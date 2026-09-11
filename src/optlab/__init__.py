"""Optimizer Lab — intuition-building experiments for optimizers and LR schedules.

The public surface is intentionally small; each experiment script imports the
pieces it needs from the submodules below.
"""

from __future__ import annotations

import random

import numpy as np
import torch

__all__ = ["set_seed", "get_device"]


def set_seed(seed: int = 42) -> None:
    """Make a run reproducible.

    Seeds Python's ``random``, NumPy, and PyTorch, and turns on PyTorch's
    deterministic algorithms so that two runs with the same seed produce
    byte-identical logs (a hard requirement of this project).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # CPU-only project, so this is safe and cheap; it forces deterministic
    # kernels and raises loudly if a non-deterministic op sneaks in.
    torch.use_deterministic_algorithms(True)


def get_device() -> torch.device:
    """Return CUDA if present, else CPU. Defaults keep everything CPU-feasible."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
