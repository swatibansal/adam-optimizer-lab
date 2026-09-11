"""Experiment 4 — Cosine vs. WSD schedule.

Train TinyLM(width=256) twice on identical data with identical randomness, differing
only in the learning-rate schedule. Stop both at step 200 first (where cosine has
already decayed but WSD is still flat), then run both to completion at step 300.

Outputs:
    figures/exp4_cosine_vs_wsd.png   (a) LR schedules, (b) loss curves
    runs/exp4/summary.json           loss at 200 and 300 for each, and which was lower
    runs/exp4/losses.csv             per-step loss and lr for both schedules (full runs)
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass
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
    peak_lr: float = 3e-3  # peak learning rate
    weight_decay: float = 0.01  # AdamW weight decay
    wsd_decay_start: int = 240  # WSD holds at peak until here, then decays to floor
    stop_at: int = 200  # first-pass early stop for the head-to-head comparison

    @property
    def floor_lr(self) -> float:
        return self.peak_lr / 10.0


def train(
    cfg: Config,
    schedule: Callable[[int], float],
    run_steps: int,
    seed: int,
) -> tuple[list[int], list[float], list[float]]:
    """Train from a fresh (seeded) model for ``run_steps`` steps.

    Returns (steps, lrs, losses); loss[i] is the training loss on the batch at
    that step, recorded before the optimizer update.
    """
    device = get_device()
    set_seed(seed)  # identical initialization for every run
    model = TinyLM(cfg.width, cfg.depth, cfg.vocab, cfg.seq_len).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.peak_lr, weight_decay=cfg.weight_decay)

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stop_at", type=int, default=200)
    args = parser.parse_args()

    cfg = Config(stop_at=args.stop_at)

    def cosine_sched(step: int) -> float:
        return cosine(step, cfg.total, cfg.peak_lr, cfg.warmup, cfg.floor_lr)

    def wsd_sched(step: int) -> float:
        return wsd(step, cfg.total, cfg.peak_lr, cfg.warmup, cfg.wsd_decay_start, cfg.floor_lr)

    # First pass: stop both at stop_at.
    steps_stop, _, loss_cos_stop = train(cfg, cosine_sched, cfg.stop_at, args.seed)
    _, _, loss_wsd_stop = train(cfg, wsd_sched, cfg.stop_at, args.seed)

    # Second pass: run both to completion.
    steps_full, lr_cos, loss_cos_full = train(cfg, cosine_sched, cfg.total, args.seed)
    _, lr_wsd, loss_wsd_full = train(cfg, wsd_sched, cfg.total, args.seed)

    loss_cos_200 = loss_cos_stop[-1]
    loss_wsd_200 = loss_wsd_stop[-1]
    loss_cos_300 = loss_cos_full[-1]
    loss_wsd_300 = loss_wsd_full[-1]

    caption = (
        f"Top: the two learning-rate schedules. Bottom: training loss, solid up to the "
        f"stop at step {cfg.stop_at} (dashed = full 300-step runs). At step {cfg.stop_at} "
        "cosine has already decayed while WSD is still at full speed; the WSD drop only "
        "shows up in its final decay."
    )
    FIGURES.mkdir(parents=True, exist_ok=True)
    plot_exp4(
        steps_stop,
        lr_cos,
        lr_wsd,
        loss_cos_stop,
        loss_wsd_stop,
        steps_full,
        loss_cos_full,
        loss_wsd_full,
        cfg.stop_at,
        FIGURES / "exp4_cosine_vs_wsd.png",
        caption,
    )

    # Per-step CSV for the full runs.
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
        "loss_at_200": {"cosine": loss_cos_200, "wsd": loss_wsd_200},
        "loss_at_300": {"cosine": loss_cos_300, "wsd": loss_wsd_300},
        "lower_at_200": "cosine" if loss_cos_200 < loss_wsd_200 else "wsd",
        "lower_at_300": "cosine" if loss_cos_300 < loss_wsd_300 else "wsd",
    }
    path = write_summary("exp4", summary)

    print(
        f"loss @200  cosine={loss_cos_200:.4f}  wsd={loss_wsd_200:.4f}"
        f"  -> lower: {summary['lower_at_200']}"
    )
    print(
        f"loss @300  cosine={loss_cos_300:.4f}  wsd={loss_wsd_300:.4f}"
        f"  -> lower: {summary['lower_at_300']}"
    )
    print(f"wrote {path}")
    print(f"wrote {FIGURES / 'exp4_cosine_vs_wsd.png'}")


if __name__ == "__main__":
    main()
