"""Synthetic next-token data — deterministic, seeded, nothing downloaded.

The task is **copy-with-offset**. Each sequence begins with ``offset`` random
tokens; every later token is a copy of the token ``offset`` positions earlier::

    seq[i] = seq[i - offset]      for i >= offset

The model is trained on standard next-token prediction (predict ``seq[i+1]`` from
``seq[:i+1]``). Because ``seq[i+1] == seq[i+1-offset]`` sits inside the context,
the task is solvable by learning to attend a fixed ``offset`` steps back — so the
loss is meaningful (a random model scores ``ln(vocab)``) and drops as the model
learns the copy rule.

Batches are produced by :func:`make_batch`, which is a pure function of
``(step, ...)``: step *k* always yields the same batch regardless of call order,
so two runs with the same seed see identical data in an identical order.
"""

from __future__ import annotations

import torch


def make_batch(
    step: int,
    batch_size: int,
    seq_len: int,
    vocab: int,
    seed: int,
    offset: int = 3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(x, y)`` LongTensors of shape ``(batch_size, seq_len)``.

    ``x`` is the input token ids and ``y`` is the same sequence shifted left by
    one (the next-token targets). Deterministic in ``(seed, step)``.
    """
    # A generator seeded per (seed, step) makes each step's batch independent of
    # how many batches were drawn before it.
    gen = torch.Generator()
    gen.manual_seed(seed * 1_000_003 + step)

    # Build sequences of length seq_len + 1, then split into input / target.
    full = torch.zeros(batch_size, seq_len + 1, dtype=torch.long)
    full[:, :offset] = torch.randint(0, vocab, (batch_size, offset), generator=gen)
    for i in range(offset, seq_len + 1):
        full[:, i] = full[:, i - offset]

    x = full[:, :-1].contiguous()
    y = full[:, 1:].contiguous()
    return x, y
