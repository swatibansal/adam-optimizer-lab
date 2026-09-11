"""Width-scaling helpers for the µP ("mu-P") check in Experiment 5.

µP is a recipe for building models so that the best learning rate does not drift
as the model gets wider. The two pieces used here are deliberately minimal:

1. **Init variance:** hidden layers are initialized with variance proportional to
   ``1 / width`` (so their standard deviation scales like ``1 / sqrt(width)``).
2. **Per-layer learning rate:** hidden layers get their LR multiplied by
   ``base_width / width``; the embedding and (tied) output head keep the base LR.

Everything is expressed relative to ``base_width`` so that at ``width ==
base_width`` the model is identical to the standard parameterization.
"""

from __future__ import annotations

import math

import torch.nn as nn

from .tiny_model import TinyLM, group_of

# Which logging groups count as "hidden" (matrix-like) layers that µP rescales.
_HIDDEN_SUFFIXES = (".attn", ".mlp")


def _is_hidden_group(group: str) -> bool:
    return group.endswith(_HIDDEN_SUFFIXES)


def hidden_lr_multiplier(width: int, base_width: int) -> float:
    """µP learning-rate multiplier for hidden layers: ``base_width / width``."""
    return base_width / width


def apply_mup_init(model: TinyLM, base_width: int, base_std: float = 0.02) -> None:
    """Rescale hidden-layer weights in place so their init variance ~ ``1 / width``.

    Hidden ``Linear`` weights get standard deviation ``base_std * sqrt(base_width /
    width)``; embeddings, biases, and LayerNorms are left as the standard init put
    them. At ``width == base_width`` this is a no-op.
    """
    std = base_std * math.sqrt(base_width / model.width)
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and _is_hidden_group_from_module_name(name):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)


def _is_hidden_group_from_module_name(module_name: str) -> bool:
    # module_name looks like "blocks.0.attn.q" / "blocks.0.mlp.fc1".
    return ".attn" in module_name or ".mlp" in module_name


def build_param_groups(
    model: TinyLM,
    base_lr: float,
    base_width: int,
    weight_decay: float,
    mup: bool,
) -> list[dict]:
    """Build AdamW parameter groups, one per logging group, with per-group LR.

    With ``mup=False`` every group uses ``base_lr`` (standard parameterization).
    With ``mup=True`` hidden groups use ``base_lr * base_width / width`` and the
    embedding/head groups keep ``base_lr``.
    """
    grouped = model.grouped_parameters()
    mult = hidden_lr_multiplier(model.width, base_width) if mup else 1.0

    param_groups: list[dict] = []
    for group_name, params in grouped.items():
        if not params:
            continue
        lr = base_lr * mult if (mup and _is_hidden_group(group_name)) else base_lr
        param_groups.append(
            {
                "params": params,
                "lr": lr,
                "weight_decay": weight_decay,
                "name": group_name,
            }
        )
    return param_groups


def lr_multipliers(model: TinyLM, base_width: int, mup: bool) -> dict[str, float]:
    """Return the LR multiplier applied to each group (handy for tests/logging)."""
    mult = hidden_lr_multiplier(model.width, base_width) if mup else 1.0
    out: dict[str, float] = {}
    for group_name in model.group_names():
        out[group_name] = mult if (mup and _is_hidden_group(group_name)) else 1.0
    return out


# Re-exported for callers that want to classify a raw parameter name.
__all__ = [
    "hidden_lr_multiplier",
    "apply_mup_init",
    "build_param_groups",
    "lr_multipliers",
    "group_of",
]
