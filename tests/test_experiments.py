"""Light tests for experiment-level logic that doesn't require training."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

from exp2_bias_correction import Config, step_size_series, steps_until_negligible  # noqa: E402


def test_uncorrected_step_starts_larger() -> None:
    cfg = Config()
    on, _ = step_size_series(cfg, bias_correction=True, n=1)
    off, _ = step_size_series(cfg, bias_correction=False, n=1)
    # The uncorrected step is larger at step 1 (v is suppressed more than m early).
    assert off[0] > on[0]


def test_difference_becomes_negligible_on_beta2_timescale() -> None:
    cfg = Config()
    step, ratios = steps_until_negligible(cfg)
    assert step is not None
    # The gap closes on the ~1/(1-b2) ≈ 1000-step memory timescale, far beyond the
    # 20 steps that get plotted.
    assert 1000 < step < cfg.max_search_steps
    assert abs(ratios[step - 1] - 1.0) <= cfg.tol_negligible
    # It has NOT yet become negligible at the end of the plotted window.
    assert abs(ratios[cfg.steps - 1] - 1.0) > cfg.tol_negligible
