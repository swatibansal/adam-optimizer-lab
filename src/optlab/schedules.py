"""Learning-rate schedules as pure functions: ``step -> lr``.

Two schedules are compared in Experiment 4:

* ``cosine`` — after a linear warm-up, decay smoothly to a floor by the last step.
* ``wsd`` — Warmup, Stable, Decay: warm up, hold at the peak, then decay linearly
  to the floor over the final stretch.

A shared ``linear_warmup`` helper (used by Experiment 3) ramps from 0 to the peak
over the warm-up window. All functions are pure and unit-tested in
``tests/test_schedules.py``.

Convention: ``step`` counts optimizer steps starting at 1. At ``step == warmup``
the learning rate reaches ``peak``; at ``step == total`` it reaches ``floor``.
"""

from __future__ import annotations

import math


def linear_warmup(step: int, peak: float, warmup: int) -> float:
    """Linear ramp from 0 at step 0 to ``peak`` at ``step == warmup`` (constant after)."""
    if warmup <= 0:
        return peak
    if step >= warmup:
        return peak
    return peak * step / warmup


def cosine(step: int, total: int, peak: float, warmup: int, floor: float) -> float:
    """Warm up to ``peak``, then cosine-decay to ``floor`` at ``step == total``."""
    if step <= warmup:
        return linear_warmup(step, peak, warmup)
    if step >= total:
        return floor
    progress = (step - warmup) / (total - warmup)  # 0 -> 1 across the decay window
    return floor + 0.5 * (peak - floor) * (1.0 + math.cos(math.pi * progress))


def wsd(
    step: int,
    total: int,
    peak: float,
    warmup: int,
    decay_start: int,
    floor: float,
) -> float:
    """Warmup, Stable, Decay.

    Warm up to ``peak`` over ``warmup`` steps, hold at ``peak`` until
    ``decay_start``, then decay linearly to ``floor`` at ``step == total``.
    """
    if step <= warmup:
        return linear_warmup(step, peak, warmup)
    if step <= decay_start:
        return peak
    if step >= total:
        return floor
    progress = (step - decay_start) / (total - decay_start)  # 0 -> 1 across the decay
    return peak + (floor - peak) * progress
