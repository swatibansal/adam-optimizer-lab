"""Experiment 1 — Adam, computed by hand.

Walk a single scalar weight through five hand-picked gradients using the pure-Python
:func:`optlab.adam_by_hand.adam_step`, printing every intermediate number, and
verify the result matches ``torch.optim.Adam`` to nine decimal places.

Outputs:
    runs/exp1/steps.csv   columns: t, g, m, v, m_hat, v_hat, step_size, w
    stdout                the same table, formatted for a human
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from optlab import set_seed  # noqa: E402
from optlab.adam_by_hand import adam_step  # noqa: E402
from optlab.logging_utils import CsvLogger, run_path  # noqa: E402


@dataclass
class Config:
    w0: float = 1.0  # starting weight
    # Five gradients chosen to oscillate (up, down, up, down, up).
    gradients: list[float] = field(default_factory=lambda: [0.5, -0.4, 0.4, -0.5, 0.5])
    lr: float = 0.01  # base step size
    b1: float = 0.9  # beta1: how much of the gradient-direction memory to keep
    b2: float = 0.999  # beta2: how much of the gradient-size memory to keep
    eps: float = 1e-8  # small constant that keeps the division safe
    tol: float = 1e-9  # required agreement with torch.optim.Adam


def run(cfg: Config) -> list[dict]:
    """Run the hand-rolled Adam and return one dict per step."""
    w, m, v = cfg.w0, 0.0, 0.0
    rows: list[dict] = []
    for t, g in enumerate(cfg.gradients, start=1):
        w_new, m, v, m_hat, v_hat = adam_step(
            w, g, m, v, t, cfg.lr, cfg.b1, cfg.b2, cfg.eps, bias_correction=True
        )
        rows.append(
            {
                "t": t,
                "g": g,
                "m": m,
                "v": v,
                "m_hat": m_hat,
                "v_hat": v_hat,
                "step_size": w - w_new,
                "w": w_new,
            }
        )
        w = w_new
    return rows


def torch_reference(cfg: Config) -> list[float]:
    """Feed the same gradients to torch.optim.Adam; return the weight after each step."""
    weight = torch.tensor([cfg.w0], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam(
        [weight], lr=cfg.lr, betas=(cfg.b1, cfg.b2), eps=cfg.eps, weight_decay=0.0
    )
    out: list[float] = []
    for g in cfg.gradients:
        opt.zero_grad()
        weight.grad = torch.tensor([g], dtype=torch.float64)
        opt.step()
        out.append(weight.item())
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)

    cfg = Config()
    rows = run(cfg)
    torch_weights = torch_reference(cfg)

    # Verify against PyTorch at every step.
    max_diff = 0.0
    for row, w_torch in zip(rows, torch_weights, strict=True):
        max_diff = max(max_diff, abs(row["w"] - w_torch))
    assert max_diff < cfg.tol, f"hand vs torch disagree by {max_diff:g} (tol {cfg.tol:g})"

    # Write CSV.
    columns = ["t", "g", "m", "v", "m_hat", "v_hat", "step_size", "w"]
    with CsvLogger(run_path("exp1", "steps.csv"), columns) as log:
        for row in rows:
            log.write(row)

    # Print a human-readable table.
    header = (
        f"{'t':>2} {'g':>7} {'m':>10} {'v':>12} {'m_hat':>10} {'v_hat':>12} {'step':>12} {'w':>10}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['t']:>2} {row['g']:>7.3f} {row['m']:>10.6f} {row['v']:>12.8f} "
            f"{row['m_hat']:>10.6f} {row['v_hat']:>12.8f} "
            f"{row['step_size']:>12.8f} {row['w']:>10.6f}"
        )
    print(f"\nMax |w_hand - w_torch| over all steps: {max_diff:.2e}  (tolerance {cfg.tol:g})")
    print("PASS: hand-rolled Adam matches torch.optim.Adam.")


if __name__ == "__main__":
    main()
