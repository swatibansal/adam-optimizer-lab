"""Experiment 2 — Bias correction on vs. off.

Run Adam with a constant gradient of +0.1, once with bias correction and once
without, and show why m_hat / v_hat exist. Plot the first 20 steps both ways, and
report the number of steps after which the difference stops mattering.

"Stops mattering" is defined precisely: the first step at which the uncorrected step
size is within ``tol`` (relative) of the corrected step size and stays there. Because
the ratio approaches 1 on the timescale of the second-moment memory (~1/(1-b2)), this
takes far longer than the 20 steps shown in the plot — so we search out to
``max_search_steps`` to find it.

Outputs:
    figures/exp2_bias_correction.png   two panels: step size, weight trajectory
    runs/exp2/summary.json             step-1 sizes, ratio, and the "stops-mattering" step
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from optlab import set_seed  # noqa: E402
from optlab.adam_by_hand import adam_step  # noqa: E402
from optlab.logging_utils import write_summary  # noqa: E402
from optlab.plotting import plot_exp2  # noqa: E402

FIGURES = Path(__file__).resolve().parents[1] / "figures"


@dataclass
class Config:
    steps: int = 20  # number of Adam steps to plot
    grad: float = 0.1  # constant gradient fed every step
    w0: float = 1.0  # starting weight
    lr: float = 0.01  # base step size
    b1: float = 0.9  # beta1
    b2: float = 0.999  # beta2
    eps: float = 1e-8  # epsilon
    tol_negligible: float = 0.05  # "stops mattering" = within 5% relative step size
    max_search_steps: int = 4000  # how far to search for the stops-mattering step


def step_size_series(cfg: Config, bias_correction: bool, n: int) -> tuple[list[float], list[float]]:
    """Return (step_sizes, weights) over ``n`` steps for one correction setting."""
    w, m, v = cfg.w0, 0.0, 0.0
    step_sizes: list[float] = []
    weights: list[float] = []
    for t in range(1, n + 1):
        w_new, m, v, _, _ = adam_step(
            w, cfg.grad, m, v, t, cfg.lr, cfg.b1, cfg.b2, cfg.eps, bias_correction
        )
        step_sizes.append(w - w_new)
        weights.append(w_new)
        w = w_new
    return step_sizes, weights


def steps_until_negligible(cfg: Config) -> tuple[int | None, list[float]]:
    """First step where |uncorrected/corrected - 1| <= tol; also return the ratio series.

    The ratio starts above 1, climbs, then decays monotonically toward 1, so the first
    step inside the band is a stable crossing (it does not re-open afterward).
    """
    on, _ = step_size_series(cfg, True, cfg.max_search_steps)
    off, _ = step_size_series(cfg, False, cfg.max_search_steps)
    ratios = [o / c for o, c in zip(off, on, strict=True)]
    for t, r in enumerate(ratios, start=1):
        if abs(r - 1.0) <= cfg.tol_negligible:
            return t, ratios
    return None, ratios


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)

    cfg = Config()
    steps = list(range(1, cfg.steps + 1))
    size_on, w_on = step_size_series(cfg, True, cfg.steps)
    size_off, w_off = step_size_series(cfg, False, cfg.steps)

    step1_on = size_on[0]
    step1_off = size_off[0]
    ratio = step1_off / step1_on
    peak_ratio = max(size_off) / step1_on

    negligible_step, ratios = steps_until_negligible(cfg)
    ratio_at_20 = ratios[cfg.steps - 1]
    pct = int(round(cfg.tol_negligible * 100))
    neg_txt = f"step ~{negligible_step}" if negligible_step else f">{cfg.max_search_steps}"

    caption = (
        "With correction on (blue), every step is the intended size from step 1. With it off "
        f"(orange), the step starts {ratio:.2f}x too large and climbs to ~{peak_ratio:.1f}x near "
        f"step 12; the two agree to within {pct}% only around {neg_txt} — the ~1/(1-b2) timescale "
        "of the second-moment memory, far beyond these 20 steps."
    )

    FIGURES.mkdir(parents=True, exist_ok=True)
    plot_exp2(
        steps,
        size_on,
        size_off,
        w_on,
        w_off,
        FIGURES / "exp2_bias_correction.png",
        caption,
    )

    summary = {
        "step1_step_size_corrected": step1_on,
        "step1_step_size_uncorrected": step1_off,
        "ratio_uncorrected_over_corrected": ratio,
        "step_size_ratio_at_step20": ratio_at_20,
        "difference_negligible_after_step": negligible_step,
        "negligible_tolerance": cfg.tol_negligible,
        "note": (
            "Uncorrected steps start LARGER than corrected ones (ratio > 1), not smaller: early "
            "on v is suppressed toward zero more strongly than m, so the uncorrected step is "
            f"inflated. It is still ~{ratio_at_20:.1f}x too large at step 20 and only closes to "
            f"within {int(round(cfg.tol_negligible * 100))}% around step {negligible_step}, which "
            "tracks the ~1/(1-b2) ≈ 1000-step memory timescale rather than a handful of steps."
        ),
    }
    path = write_summary("exp2", summary)

    print(f"step-1 step size  corrected: {step1_on:.8f}")
    print(f"step-1 step size uncorrected: {step1_off:.8f}")
    print(f"ratio (uncorrected / corrected) at step 1: {ratio:.4f}")
    print(f"ratio at step 20: {ratio_at_20:.4f}")
    print(f"difference negligible (within {pct}%) after step: {negligible_step}")
    print(f"wrote {path}")
    print(f"wrote {FIGURES / 'exp2_bias_correction.png'}")


if __name__ == "__main__":
    main()
