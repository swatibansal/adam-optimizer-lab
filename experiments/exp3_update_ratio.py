"""Experiment 3 — Weight-to-update ratio and warm-up.

Train TinyLM(width=256) with AdamW for 300 steps using a linear warm-up over the
first 30 steps, then constant. After every optimizer step, measure how much each
parameter group actually moved: ‖Δw‖₂ / ‖w‖₂.

Outputs:
    runs/exp3/update_ratio.csv   columns: step, group, ratio, lr
    runs/exp3/summary.json       warmup_effect_ends_step and per-group ratios at step 60
    figures/exp3_update_ratio.png
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from optlab import get_device, set_seed  # noqa: E402
from optlab.data import make_batch  # noqa: E402
from optlab.logging_utils import CsvLogger, run_path, write_summary  # noqa: E402
from optlab.plotting import plot_exp3  # noqa: E402
from optlab.schedules import linear_warmup  # noqa: E402
from optlab.tiny_model import TinyLM, group_of  # noqa: E402

FIGURES = Path(__file__).resolve().parents[1] / "figures"


@dataclass
class Config:
    width: int = 256  # model width
    depth: int = 2  # transformer blocks
    vocab: int = 512  # vocabulary size
    seq_len: int = 64  # sequence length
    steps: int = 300  # training steps
    batch: int = 32  # batch size
    warmup: int = 30  # linear warm-up length (steps)
    peak_lr: float = 3e-3  # learning rate after warm-up
    weight_decay: float = 0.01  # AdamW weight decay
    reference_step: int = 60  # ratios are compared against their value here
    tol_frac: float = 0.10  # "stabilized" = within 10% of the reference value


def group_ratios(model: TinyLM, before: dict[str, torch.Tensor]) -> dict[str, float]:
    """Compute ‖Δw‖₂ / ‖w‖₂ per group from a pre-step snapshot ``before``."""
    dsq: dict[str, float] = {}
    wsq: dict[str, float] = {}
    for name, param in model.named_parameters():
        g = group_of(name)
        delta = param.detach() - before[name]
        dsq[g] = dsq.get(g, 0.0) + float((delta * delta).sum())
        wsq[g] = wsq.get(g, 0.0) + float((param.detach() * param.detach()).sum())
    return {g: (dsq[g] ** 0.5) / (wsq[g] ** 0.5) for g in dsq}


def find_stabilized_step(
    steps: list[int],
    ratios_by_group: dict[str, list[float]],
    reference_step: int,
    tol_frac: float,
) -> int | None:
    """Earliest step where every group's ratio has settled to within tol_frac of its
    reference-step value.

    The reference (step 60) is taken as the "settled" value each group is heading
    toward. We scan from the very first step so that warm-up steps — where the
    ratios are still ramping and sit well outside the band — are correctly excluded,
    and we return the first step at which all groups are simultaneously inside it.
    """
    ref_idx = steps.index(reference_step)
    refs = {g: r[ref_idx] for g, r in ratios_by_group.items()}
    for i, step in enumerate(steps):
        if all(abs(ratios_by_group[g][i] - refs[g]) <= tol_frac * refs[g] for g in refs):
            return step
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)

    cfg = Config()
    device = get_device()
    model = TinyLM(cfg.width, cfg.depth, cfg.vocab, cfg.seq_len).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.peak_lr, weight_decay=cfg.weight_decay)

    steps: list[int] = []
    ratios_by_group: dict[str, list[float]] = {g: [] for g in model.group_names()}

    log = CsvLogger(run_path("exp3", "update_ratio.csv"), ["step", "group", "ratio", "lr"])
    for step in range(1, cfg.steps + 1):
        lr = linear_warmup(step, cfg.peak_lr, cfg.warmup)
        for pg in opt.param_groups:
            pg["lr"] = lr

        x, y = make_batch(step, cfg.batch, cfg.seq_len, cfg.vocab, args.seed)
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, cfg.vocab), y.reshape(-1))
        opt.zero_grad()
        loss.backward()

        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        opt.step()

        ratios = group_ratios(model, before)
        steps.append(step)
        for g, r in ratios.items():
            ratios_by_group[g].append(r)
            log.write({"step": step, "group": g, "ratio": r, "lr": lr})
    log.close()

    stabilized = find_stabilized_step(steps, ratios_by_group, cfg.reference_step, cfg.tol_frac)

    caption = (
        "Each line is one layer group's move-size relative to its own size, per step "
        "(log scale). Warm-up is shaded; the dashed line marks where every group has settled "
        "to within 10% of its step-60 value. Healthy is ~1e-3 per step (green dots): much "
        "larger means the layer is thrown around, much smaller means it barely learns."
    )
    FIGURES.mkdir(parents=True, exist_ok=True)
    plot_exp3(
        steps,
        ratios_by_group,
        cfg.warmup,
        stabilized,
        FIGURES / "exp3_update_ratio.png",
        caption,
    )

    ref_idx = steps.index(cfg.reference_step)
    summary = {
        "warmup_effect_ends_step": stabilized,
        "warmup_steps": cfg.warmup,
        "ratios_at_reference_step": {g: ratios_by_group[g][ref_idx] for g in ratios_by_group},
        "reference_step": cfg.reference_step,
        "final_ratios": {g: ratios_by_group[g][-1] for g in ratios_by_group},
    }
    path = write_summary("exp3", summary)

    print(f"warmup_effect_ends_step: {stabilized}")
    print(f"wrote {path}")
    print(f"wrote {FIGURES / 'exp3_update_ratio.png'}")


if __name__ == "__main__":
    main()
