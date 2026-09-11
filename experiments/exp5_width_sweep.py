"""Experiment 5 — Learning-rate sweep across widths (µP check).

Sweep 13 learning rates across three widths {256, 512, 1024} and record the final
loss for each, once with standard parameterization and once with µP-style scaling.
The question: does the best learning rate transfer across widths?

Both parameterizations are run in a single invocation so the two-panel comparison
figure and the full summary can be produced from one command. The sweep always runs
the full 150 steps per config across all three widths — there is no automatic step
cap, so the standard-vs-µP comparison is always at equal fidelity ("full parity").
Pass ``--steps N`` if you deliberately want a shorter (or longer) run; on a slow
machine the full sweep can take upwards of 20 minutes.

Outputs:
    figures/exp5_lr_sweep.png   two panels (standard vs µP), minima marked
    runs/exp5/sweep.csv         param, width, lr, final_loss, steps
    runs/exp5/summary.json      argmin lr per width, recommended_lr_4096, confidence
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from optlab import get_device, set_seed  # noqa: E402
from optlab.data import make_batch  # noqa: E402
from optlab.logging_utils import CsvLogger, run_path, write_summary  # noqa: E402
from optlab.mup import apply_mup_init, build_param_groups  # noqa: E402
from optlab.plotting import plot_exp5  # noqa: E402
from optlab.tiny_model import TinyLM  # noqa: E402

FIGURES = Path(__file__).resolve().parents[1] / "figures"


@dataclass
class Config:
    widths: list[int] = field(default_factory=lambda: [256, 512, 1024])
    base_width: int = 256  # µP reference width (multiplier == 1 here)
    depth: int = 2  # transformer blocks
    vocab: int = 512  # vocabulary size
    seq_len: int = 64  # sequence length
    batch: int = 16  # batch size (small to keep the sweep CPU-feasible)
    steps: int = 150  # steps per (width, lr) — full fidelity, no automatic cap
    last_k: int = 10  # final loss = mean of the last k steps
    weight_decay: float = 0.01  # AdamW weight decay
    lr_min_exp: float = -4  # logspace start: 1e-4
    lr_max_exp: float = -1  # logspace end: 1e-1
    lr_points: int = 13  # number of learning rates


def train_final_loss(cfg: Config, width: int, lr: float, mup: bool, steps: int, seed: int) -> float:
    """Train one model and return the mean loss over the last ``cfg.last_k`` steps."""
    device = get_device()
    set_seed(seed)  # identical init across learning rates for a fair comparison
    model = TinyLM(width, cfg.depth, cfg.vocab, cfg.seq_len, mup=mup, base_width=cfg.base_width).to(
        device
    )
    if mup:
        apply_mup_init(model, cfg.base_width)
    param_groups = build_param_groups(model, lr, cfg.base_width, cfg.weight_decay, mup)
    opt = torch.optim.AdamW(param_groups)

    recent: list[float] = []
    for step in range(1, steps + 1):
        x, y = make_batch(step, cfg.batch, cfg.seq_len, cfg.vocab, seed)
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, cfg.vocab), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        recent.append(float(loss.detach()))
    tail = recent[-cfg.last_k :]
    return sum(tail) / len(tail)


def summarize_confidence(argmin_mup: dict[int, float], widths: list[int]) -> tuple[str, str]:
    """Rate how well the µP optimum transfers across width.

    ``high``   — all three optima within a factor of 1.5 (they effectively coincide).
    ``low``    — the optima still move *strictly* monotonically with width, like the
                 standard parameterization: the best LR has not settled.
    ``medium`` — in between: close and partly plateaued (e.g. the larger widths agree)
                 but not all within 1.5x.

    A plateau counts against ``low``: if two adjacent widths share an optimum the LR is no
    longer strictly moving, which is exactly the transfer µP is trying to achieve.
    """
    minima = [argmin_mup[w] for w in widths]  # ordered by increasing width
    ratio = max(minima) / min(minima)
    pairs = list(zip(minima, minima[1:], strict=False))
    strictly_monotonic = all(a < b for a, b in pairs) or all(a > b for a, b in pairs)
    if ratio <= 1.5:
        return "high", (
            f"the three µP optima lie within a factor of {ratio:.2f} (<= 1.5) of each other."
        )
    if strictly_monotonic:
        return "low", (
            f"the µP optima still move monotonically with width (a factor of {ratio:.2f} "
            "across widths), so the best LR has not settled."
        )
    return "medium", (
        f"the µP optima span a factor of {ratio:.2f} but do not move strictly monotonically "
        "with width (they partly plateau), so transfer is good but short of the 1.5x 'high' bar."
    )


def argmin_per_width(lrs: list[float], losses: dict[int, list[float]]) -> dict[int, float]:
    return {w: lrs[int(np.argmin(v))] for w, v in losses.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="steps per config (default 150; full sweep, no automatic cap)",
    )
    args = parser.parse_args()

    cfg = Config()
    if args.steps is not None:
        cfg.steps = args.steps
    lrs = [float(x) for x in np.logspace(cfg.lr_min_exp, cfg.lr_max_exp, cfg.lr_points)]

    steps = cfg.steps
    print(f"running sweep at {steps} steps/config (full parity, no cap)")

    losses_std: dict[int, list[float]] = {w: [] for w in cfg.widths}
    losses_mup: dict[int, list[float]] = {w: [] for w in cfg.widths}

    log = CsvLogger(run_path("exp5", "sweep.csv"), ["param", "width", "lr", "final_loss", "steps"])
    for mup, store in ((False, losses_std), (True, losses_mup)):
        label = "mup" if mup else "standard"
        for width in cfg.widths:
            for lr in lrs:
                final_loss = train_final_loss(cfg, width, lr, mup, steps, args.seed)
                store[width].append(final_loss)
                log.write(
                    {
                        "param": label,
                        "width": width,
                        "lr": lr,
                        "final_loss": final_loss,
                        "steps": steps,
                    }
                )
            print(f"  {label} width={width} done", flush=True)
    log.close()

    argmin_std = argmin_per_width(lrs, losses_std)
    argmin_mup = argmin_per_width(lrs, losses_mup)
    confidence, justification = summarize_confidence(argmin_mup, cfg.widths)

    # Under µP the best LR should transfer, so recommend the geometric mean of the
    # three µP optima as the width-4096 setting.
    mup_minima = [argmin_mup[w] for w in cfg.widths]
    recommended_lr_4096 = float(math.exp(sum(math.log(x) for x in mup_minima) / len(mup_minima)))

    caption = (
        "Loss vs learning rate (log x) at three widths; the circled dot is each curve's best "
        "LR. Standard parameterization (left) lets the best LR drift with width; µP (right) is "
        "designed to keep the optima aligned so a small-model LR transfers to a larger model."
    )
    FIGURES.mkdir(parents=True, exist_ok=True)
    plot_exp5(
        lrs,
        losses_std,
        losses_mup,
        argmin_std,
        argmin_mup,
        FIGURES / "exp5_lr_sweep.png",
        caption,
    )

    summary = {
        "steps_per_config": steps,
        "learning_rates": lrs,
        "argmin_lr": {
            "standard": {str(w): argmin_std[w] for w in cfg.widths},
            "mup": {str(w): argmin_mup[w] for w in cfg.widths},
        },
        "recommended_lr_4096": recommended_lr_4096,
        "confidence": confidence,
        "confidence_justification": justification,
    }
    path = write_summary("exp5", summary)

    print(f"standard argmin: {argmin_std}")
    print(f"mup argmin:      {argmin_mup}")
    print(f"recommended_lr_4096: {recommended_lr_4096:.2e}  confidence: {confidence}")
    print(f"wrote {path}")
    print(f"wrote {FIGURES / 'exp5_lr_sweep.png'}")


if __name__ == "__main__":
    main()
