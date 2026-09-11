"""Unit tests for the learning-rate schedules."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from optlab.schedules import cosine, linear_warmup, wsd

TOTAL = 300
PEAK = 3e-3
WARMUP = 15
FLOOR = PEAK / 10
DECAY_START = 240


def test_linear_warmup_endpoints() -> None:
    assert linear_warmup(0, PEAK, WARMUP) == 0.0
    assert linear_warmup(WARMUP, PEAK, WARMUP) == pytest.approx(PEAK)
    assert linear_warmup(WARMUP + 50, PEAK, WARMUP) == pytest.approx(PEAK)
    # Halfway through warm-up is half the peak.
    assert linear_warmup(WARMUP // 2, PEAK, WARMUP) == pytest.approx(PEAK * (WARMUP // 2) / WARMUP)


def test_cosine_endpoints_and_monotonic_decay() -> None:
    assert cosine(WARMUP, TOTAL, PEAK, WARMUP, FLOOR) == pytest.approx(PEAK)
    assert cosine(TOTAL, TOTAL, PEAK, WARMUP, FLOOR) == pytest.approx(FLOOR)
    # Strictly decreasing across the decay window.
    vals = [cosine(s, TOTAL, PEAK, WARMUP, FLOOR) for s in range(WARMUP, TOTAL + 1)]
    assert all(a >= b - 1e-12 for a, b in zip(vals, vals[1:], strict=False))
    # Never dips below the floor or above the peak.
    assert min(vals) >= FLOOR - 1e-12
    assert max(vals) <= PEAK + 1e-12


def test_cosine_midpoint_is_halfway() -> None:
    mid = (WARMUP + TOTAL) / 2
    # cos(pi/2) = 0, so the midpoint LR is ~floor + half the range (integer step 157
    # is just shy of the true 157.5 midpoint, hence the loose tolerance).
    assert cosine(int(mid), TOTAL, PEAK, WARMUP, FLOOR) == pytest.approx(
        FLOOR + 0.5 * (PEAK - FLOOR), rel=1e-2
    )


def test_wsd_stable_then_decay() -> None:
    assert wsd(WARMUP, TOTAL, PEAK, WARMUP, DECAY_START, FLOOR) == pytest.approx(PEAK)
    # Flat at the peak through the stable phase.
    for s in range(WARMUP, DECAY_START + 1):
        assert wsd(s, TOTAL, PEAK, WARMUP, DECAY_START, FLOOR) == pytest.approx(PEAK)
    # Hits the floor exactly at the end.
    assert wsd(TOTAL, TOTAL, PEAK, WARMUP, DECAY_START, FLOOR) == pytest.approx(FLOOR)
    # Linear decay: the midpoint of the decay window is halfway between peak and floor.
    mid = (DECAY_START + TOTAL) // 2
    assert wsd(mid, TOTAL, PEAK, WARMUP, DECAY_START, FLOOR) == pytest.approx(
        (PEAK + FLOOR) / 2, rel=1e-2
    )


def test_wsd_decay_is_monotonic() -> None:
    vals = [wsd(s, TOTAL, PEAK, WARMUP, DECAY_START, FLOOR) for s in range(DECAY_START, TOTAL + 1)]
    assert all(a >= b - 1e-12 for a, b in zip(vals, vals[1:], strict=False))
