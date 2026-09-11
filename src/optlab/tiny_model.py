"""The tiny language model shared by Experiments 3-5.

``TinyLM`` is a minimal decoder-only transformer:

    token embedding (+ learned positional embedding)
      -> depth x [ LayerNorm -> causal self-attention (single head)
                   LayerNorm -> MLP with 4x expansion ]
      -> LayerNorm
      -> tied output head (logits = hidden @ token_embedding.weight.T)

Width is a constructor argument so Experiment 5 can instantiate 256 / 512 / 1024
with no other code changes. Attention is written out by hand (rather than using
``scaled_dot_product_attention``) so it plays nicely with deterministic mode and
so every operation is inspectable.

Parameters are grouped by name via :func:`group_of` so per-layer logging works:
``embed``, ``block{i}.attn``, ``block{i}.mlp``, ``head``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """Single-head causal self-attention (single head is plenty for a tiny model).

    ``attn_scale`` multiplies the raw q·k scores. Standard attention uses
    ``1/sqrt(width)``; the µP variant uses ``sqrt(base_width)/width`` so the attention
    logits stay O(1) as width grows (see :class:`TinyLM`).
    """

    def __init__(self, width: int, seq_len: int, attn_scale: float) -> None:
        super().__init__()
        self.width = width
        self.attn_scale = attn_scale
        self.q = nn.Linear(width, width)
        self.k = nn.Linear(width, width)
        self.v = nn.Linear(width, width)
        self.proj = nn.Linear(width, width)
        # Lower-triangular causal mask, registered so it moves with .to(device).
        mask = torch.tril(torch.ones(seq_len, seq_len))
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, seq, _ = x.shape
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        scores = (q @ k.transpose(-2, -1)) * self.attn_scale
        scores = scores.masked_fill(self.mask[:seq, :seq] == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out = attn @ v
        return self.proj(out)


class MLP(nn.Module):
    """Position-wise MLP with 4x expansion and GELU."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(width, 4 * width)
        self.fc2 = nn.Linear(4 * width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    """One transformer block: pre-norm attention then pre-norm MLP, both residual."""

    def __init__(self, width: int, seq_len: int, attn_scale: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(width)
        self.attn = CausalSelfAttention(width, seq_len, attn_scale)
        self.ln2 = nn.LayerNorm(width)
        self.mlp = MLP(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyLM(nn.Module):
    """A tiny decoder-only language model with a tied output head.

    Set ``mup=True`` to enable the µP (maximal-update) forward-pass rescalings used by
    Experiment 5, expressed relative to ``base_width``:

    * attention logits are scaled by ``sqrt(base_width)/width`` instead of the standard
      ``1/sqrt(width)``, keeping them O(1) as width grows (equals the standard scale at
      ``width == base_width``);
    * the readout logits are multiplied by ``base_width/width`` so the output magnitude
      stays O(1) as width grows.

    The complementary µP pieces — hidden-layer init variance and per-layer learning
    rates — live in :mod:`optlab.mup`. With ``mup=False`` this is an ordinary model and
    Experiments 3-4 are unaffected.
    """

    def __init__(
        self,
        width: int,
        depth: int = 2,
        vocab: int = 512,
        seq_len: int = 64,
        mup: bool = False,
        base_width: int = 256,
    ) -> None:
        super().__init__()
        self.width = width
        self.depth = depth
        self.vocab = vocab
        self.seq_len = seq_len
        self.mup = mup
        self.base_width = base_width

        if mup:
            self.attn_scale = math.sqrt(base_width) / width
            self.out_mult = base_width / width
        else:
            self.attn_scale = 1.0 / math.sqrt(width)
            self.out_mult = 1.0

        self.tok_embed = nn.Embedding(vocab, width)
        self.pos_embed = nn.Parameter(torch.zeros(seq_len, width))
        self.blocks = nn.ModuleList([Block(width, seq_len, self.attn_scale) for _ in range(depth)])
        self.final_ln = nn.LayerNorm(width)

        self._init_weights()

    def _init_weights(self) -> None:
        # Modest, standard init; Experiment 5's mup path rescales the hidden
        # layers on top of this.
        nn.init.normal_(self.tok_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """idx: (batch, seq) token ids -> logits: (batch, seq, vocab)."""
        _, seq = idx.shape
        x = self.tok_embed(idx) + self.pos_embed[:seq]
        for block in self.blocks:
            x = block(x)
        x = self.final_ln(x)
        # Tied output head: reuse the token embedding matrix. The µP output multiplier
        # is 1.0 in the standard model, so this is a no-op there.
        return (x @ self.tok_embed.weight.t()) * self.out_mult

    def group_names(self) -> list[str]:
        """The distinct parameter-group names, in a stable order."""
        names = ["embed"]
        for i in range(self.depth):
            names.append(f"block{i}.attn")
            names.append(f"block{i}.mlp")
        names.append("head")
        return names

    def grouped_parameters(self) -> dict[str, list[torch.nn.Parameter]]:
        """Map each group name to the list of parameters that belong to it."""
        groups: dict[str, list[torch.nn.Parameter]] = {n: [] for n in self.group_names()}
        for name, param in self.named_parameters():
            groups[group_of(name)].append(param)
        return groups


def group_of(name: str) -> str:
    """Classify a parameter's dotted name into one of the logging groups.

    ``embed``         -> token + positional embeddings
    ``block{i}.attn`` -> that block's LayerNorm-1 and attention weights
    ``block{i}.mlp``  -> that block's LayerNorm-2 and MLP weights
    ``head``          -> the final LayerNorm (the output head itself is tied to embed)
    """
    if name.startswith("tok_embed") or name.startswith("pos_embed"):
        return "embed"
    if name.startswith("blocks."):
        i = name.split(".")[1]
        if ".attn" in name or ".ln1" in name:
            return f"block{i}.attn"
        if ".mlp" in name or ".ln2" in name:
            return f"block{i}.mlp"
    if name.startswith("final_ln"):
        return "head"
    raise KeyError(f"unclassified parameter name: {name!r}")
