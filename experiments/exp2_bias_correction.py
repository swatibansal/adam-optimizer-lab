"""Experiment 2 — Bias correction on vs. off.

Run 20 Adam steps with a constant gradient of +0.1, once with bias correction and
once without, and show why m_hat / v_hat exist.

Outputs:
    figures/exp2_bias_correction.png   two panels: step size, weight trajectory
    runs/exp2/summary.json             step-1 step size per variant and their ratio
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
    steps: int = 20  # number of Adam steps
    grad: float = 0.1  # constant gradient fed every step
    w0: float = 1.0  # starting weight
    lr: float = 0.01  # base step size
    b1: float = 0.9  # beta1
    b2: float = 0.999  # beta2
    eps: float = 1e-8  # epsilon


def run(cfg: Config, bias_correction: bool) -> tuple[list[float], list[float]]:
    """Return (step_sizes, weights) over the run for one correction setting."""
    w, m, v = cfg.w0, 0.0, 0.0
    step_sizes: list[float] = []
    weights: list[float] = []
    for t in range(1, cfg.steps + 1):
        w_new, m, v, _, _ = adam_step(
            w, cfg.grad, m, v, t, cfg.lr, cfg.b1, cfg.b2, cfg.eps, bias_correction
        )
        step_sizes.append(w - w_new)
        weights.append(w_new)
        w = w_new
    return step_sizes, weights


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)

    cfg = Config()
    steps = list(range(1, cfg.steps + 1))
    size_on, w_on = run(cfg, bias_correction=True)
    size_off, w_off = run(cfg, bias_correction=False)

    step1_on = size_on[0]
    step1_off = size_off[0]
    ratio = step1_off / step1_on

    peak_ratio = max(size_off) / step1_on
    caption = (
        "With correction on (blue), every step is the intended size from step 1. With it off "
        f"(orange), the step starts {ratio:.2f}x too large and keeps climbing to "
        f"~{peak_ratio:.1f}x around step 12 before easing back — the opposite of the naive "
        "intuition that early steps are tiny."
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
        "note": (
            "Uncorrected steps start LARGER than corrected ones (ratio > 1), not smaller: "
            "the second-moment estimate v is suppressed toward zero more strongly than the "
            "first-moment estimate m early on, so the uncorrected step is inflated until both "
            "memories fill up."
        ),
    }
    path = write_summary("exp2", summary)

    print(f"step-1 step size  corrected: {step1_on:.8f}")
    print(f"step-1 step size uncorrected: {step1_off:.8f}")
    print(f"ratio (uncorrected / corrected): {ratio:.4f}")
    print(f"wrote {path}")
    print(f"wrote {FIGURES / 'exp2_bias_correction.png'}")


if __name__ == "__main__":
    main()
