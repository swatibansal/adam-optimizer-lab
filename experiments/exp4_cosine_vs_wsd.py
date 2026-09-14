"""Experiment 4 — Cosine vs. WSD schedule (both sides tuned).

Train TinyLM(width=256) under two learning-rate schedules on identical data and
randomness, stop both at step 200, and state which model to keep.

Crucially, each schedule's peak learning rate is tuned **independently** over the same
grid before the comparison — the single most common way optimizer/schedule claims fail
to replicate is a well-tuned method measured against a badly-tuned one. Cosine and WSD
prefer different peaks (WSD is still at full speed at step 200, cosine has decayed), so
comparing both at one shared LR would be unfair.

Outputs:
    figures/exp4_cosine_vs_wsd.png   (a) tuned LR schedules, (b) loss curves + zoom
    runs/exp4/summary.json           tuned peak per schedule, losses, and which to keep
    runs/exp4/losses.csv             per-step loss and lr for the tuned full runs
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from optlab import get_device, set_seed  # noqa: E402
from optlab.data import make_batch  # noqa: E402
from optlab.logging_utils import CsvLogger, run_path, write_summary  # noqa: E402
from optlab.plotting import plot_exp4  # noqa: E402
from optlab.schedules import cosine, wsd  # noqa: E402
from optlab.tiny_model import TinyLM  # noqa: E402

FIGURES = Path(__file__).resolve().parents[1] / "figures"


@dataclass
class Config:
    width: int = 256  # model width
    depth: int = 2  # transformer blocks
    vocab: int = 512  # vocabulary size
    seq_len: int = 64  # sequence length
    total: int = 300  # planned training steps (defines the schedules)
    batch: int = 32  # batch size
    warmup: int = 15  # warm-up length (steps)
    weight_decay: float = 0.01  # AdamW weight decay
    wsd_decay_start: int = 240  # WSD holds at peak until here, then decays to floor
    stop_at: int = 200  # decision point: both runs are compared here
    # Peak-LR grid each schedule is tuned over independently (floor = peak / 10).
    lr_grid: list[float] = field(
        default_factory=lambda: [1.0e-3, 1.78e-3, 3.16e-3, 5.62e-3, 1.0e-2]
    )


# A schedule factory maps a peak LR to a step -> lr function.
ScheduleFactory = Callable[[float], Callable[[int], float]]


def train(
    cfg: Config,
    schedule: Callable[[int], float],
    run_steps: int,
    seed: int,
) -> tuple[list[int], list[float], list[float]]:
    """Train from a fresh (seeded) model for ``run_steps`` steps under ``schedule``.

    Returns (steps, lrs, losses); loss[i] is the training loss on the batch at that
    step, recorded before the optimizer update. The AdamW LR is set from the schedule
    every step, so its init value is immaterial.
    """
    device = get_device()
    set_seed(seed)  # identical initialization for every run
    model = TinyLM(cfg.width, cfg.depth, cfg.vocab, cfg.seq_len).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=schedule(1), weight_decay=cfg.weight_decay)

    steps: list[int] = []
    lrs: list[float] = []
    losses: list[float] = []
    for step in range(1, run_steps + 1):
        lr = schedule(step)
        for pg in opt.param_groups:
            pg["lr"] = lr

        x, y = make_batch(step, cfg.batch, cfg.seq_len, cfg.vocab, seed)
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, cfg.vocab), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()

        steps.append(step)
        lrs.append(lr)
        losses.append(float(loss.detach()))
    return steps, lrs, losses


def tune_peak_lr(
    cfg: Config, factory: ScheduleFactory, seed: int
) -> tuple[float, dict[float, float]]:
    """Sweep the peak LR grid, training to ``stop_at`` each time.

    Returns (best_peak, {peak: loss_at_stop}). The best peak minimizes the loss at the
    decision point, so each schedule is compared at its own best setting.
    """
    grid_loss: dict[float, float] = {}
    for peak in cfg.lr_grid:
        _, _, losses = train(cfg, factory(peak), cfg.stop_at, seed)
        grid_loss[peak] = losses[-1]
    best_peak = min(grid_loss, key=lambda p: grid_loss[p])
    return best_peak, grid_loss


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stop_at", type=int, default=200)
    args = parser.parse_args()

    cfg = Config(stop_at=args.stop_at)

    def cosine_factory(peak: float) -> Callable[[int], float]:
        return lambda step: cosine(step, cfg.total, peak, cfg.warmup, peak / 10.0)

    def wsd_factory(peak: float) -> Callable[[int], float]:
        return lambda step: wsd(step, cfg.total, peak, cfg.warmup, cfg.wsd_decay_start, peak / 10.0)

    # Tune each schedule independently at the step-200 decision point.
    cos_peak, cos_grid = tune_peak_lr(cfg, cosine_factory, args.seed)
    wsd_peak, wsd_grid = tune_peak_lr(cfg, wsd_factory, args.seed)

    # Run each at its tuned peak to completion for the plot and the 300-step context.
    steps_full, lr_cos, loss_cos_full = train(cfg, cosine_factory(cos_peak), cfg.total, args.seed)
    _, lr_wsd, loss_wsd_full = train(cfg, wsd_factory(wsd_peak), cfg.total, args.seed)

    s = cfg.stop_at
    loss_cos_200, loss_wsd_200 = loss_cos_full[s - 1], loss_wsd_full[s - 1]
    loss_cos_300, loss_wsd_300 = loss_cos_full[-1], loss_wsd_full[-1]
    keep = "cosine" if loss_cos_200 < loss_wsd_200 else "wsd"
    keep_reason = (
        f"at the step-{s} decision point the {keep} model has the lower loss "
        f"({min(loss_cos_200, loss_wsd_200):.4f} vs {max(loss_cos_200, loss_wsd_200):.4f}), "
        f"with each schedule at its own tuned peak LR (cosine {cos_peak:.2e}, WSD {wsd_peak:.2e})."
    )

    caption = (
        f"Each schedule is LR-tuned independently (cosine peak {cos_peak:.1e}, WSD peak "
        f"{wsd_peak:.1e}) so the comparison is fair. Top: the tuned schedules. Bottom: loss, "
        f"solid to the step-{s} decision point (dashed = full 300), with a late-training zoom. "
        f"We keep {keep}."
    )
    FIGURES.mkdir(parents=True, exist_ok=True)
    plot_exp4(
        steps_full[:s],
        lr_cos,
        lr_wsd,
        loss_cos_full[:s],
        loss_wsd_full[:s],
        steps_full,
        loss_cos_full,
        loss_wsd_full,
        cfg.stop_at,
        FIGURES / "exp4_cosine_vs_wsd.png",
        caption,
    )

    csv_columns = ["step", "cosine_lr", "cosine_loss", "wsd_lr", "wsd_loss"]
    with CsvLogger(run_path("exp4", "losses.csv"), csv_columns) as log:
        for i, step in enumerate(steps_full):
            log.write(
                {
                    "step": step,
                    "cosine_lr": lr_cos[i],
                    "cosine_loss": loss_cos_full[i],
                    "wsd_lr": lr_wsd[i],
                    "wsd_loss": loss_wsd_full[i],
                }
            )

    summary = {
        "stop_at": cfg.stop_at,
        "both_sides_tuned": True,
        "lr_grid": cfg.lr_grid,
        "tuned_peak_lr": {"cosine": cos_peak, "wsd": wsd_peak},
        "lr_sweep_loss_at_stop": {
            "cosine": {f"{p:.2e}": cos_grid[p] for p in cfg.lr_grid},
            "wsd": {f"{p:.2e}": wsd_grid[p] for p in cfg.lr_grid},
        },
        "loss_at_200": {"cosine": loss_cos_200, "wsd": loss_wsd_200},
        "loss_at_300": {"cosine": loss_cos_300, "wsd": loss_wsd_300},
        "keep": keep,
        "keep_reason": keep_reason,
        "lower_at_300": "cosine" if loss_cos_300 < loss_wsd_300 else "wsd",
    }
    path = write_summary("exp4", summary)

    print(f"tuned peak LR  cosine={cos_peak:.2e}  wsd={wsd_peak:.2e}")
    print(f"loss @200  cosine={loss_cos_200:.4f}  wsd={loss_wsd_200:.4f}  -> keep: {keep}")
    print(f"loss @300  cosine={loss_cos_300:.4f}  wsd={loss_wsd_300:.4f}")
    print(f"wrote {path}")
    print(f"wrote {FIGURES / 'exp4_cosine_vs_wsd.png'}")


if __name__ == "__main__":
    main()
