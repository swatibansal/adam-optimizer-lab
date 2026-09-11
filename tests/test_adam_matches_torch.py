"""The hand-rolled Adam must match torch.optim.Adam to nine decimal places."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from optlab.adam_by_hand import adam_step

GRADIENTS = [0.5, -0.4, 0.4, -0.5, 0.5]
LR, B1, B2, EPS = 0.01, 0.9, 0.999, 1e-8
TOL = 1e-9


def _hand_trajectory(gradients: list[float]) -> list[float]:
    w, m, v = 1.0, 0.0, 0.0
    out: list[float] = []
    for t, g in enumerate(gradients, start=1):
        w, m, v, _, _ = adam_step(w, g, m, v, t, LR, B1, B2, EPS, bias_correction=True)
        out.append(w)
    return out


def _torch_trajectory(gradients: list[float]) -> list[float]:
    weight = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([weight], lr=LR, betas=(B1, B2), eps=EPS, weight_decay=0.0)
    out: list[float] = []
    for g in gradients:
        opt.zero_grad()
        weight.grad = torch.tensor([g], dtype=torch.float64)
        opt.step()
        out.append(weight.item())
    return out


def test_matches_torch_each_step() -> None:
    hand = _hand_trajectory(GRADIENTS)
    ref = _torch_trajectory(GRADIENTS)
    for t, (wh, wt) in enumerate(zip(hand, ref, strict=True), start=1):
        assert abs(wh - wt) < TOL, f"step {t}: hand={wh!r} torch={wt!r} diff={abs(wh - wt):g}"


def test_matches_torch_on_another_sequence() -> None:
    grads = [0.3, 0.3, -0.2, 0.7, -0.1, -0.6, 0.4]
    hand = _hand_trajectory(grads)
    ref = _torch_trajectory(grads)
    for t, (wh, wt) in enumerate(zip(hand, ref, strict=True), start=1):
        assert abs(wh - wt) < TOL, f"step {t}: diff={abs(wh - wt):g}"
