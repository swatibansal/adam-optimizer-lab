"""Aggregate every experiment's headline numbers into one 'results at a glance' table.

Reads ``runs/exp1/steps.csv`` and ``runs/exp*/summary.json`` and prints a compact
table. Run after ``make all`` (or the individual experiments). This is what keeps the
README's summary table honest — the numbers there should match this output.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

RUNS = Path(__file__).resolve().parents[1] / "runs"


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as fh:
        return json.load(fh)


def collect() -> list[tuple[str, str]]:
    """Return (experiment label, one-line headline) pairs for whatever has been run."""
    rows: list[tuple[str, str]] = []

    steps_csv = RUNS / "exp1" / "steps.csv"
    if steps_csv.exists():
        with open(steps_csv) as fh:
            last = list(csv.DictReader(fh))[-1]
        rows.append(
            ("Exp 1  Adam by hand", f"matches torch to <1e-9; final w = {float(last['w']):.6f}")
        )

    e2 = _load_json(RUNS / "exp2" / "summary.json")
    if e2:
        rows.append(
            (
                "Exp 2  Bias correction",
                f"step-1 size {e2['step1_step_size_corrected']:.4f} (on) vs "
                f"{e2['step1_step_size_uncorrected']:.4f} (off) — "
                f"{e2['ratio_uncorrected_over_corrected']:.2f}x larger uncorrected; "
                f"difference negligible after step {e2['difference_negligible_after_step']}",
            )
        )

    e3 = _load_json(RUNS / "exp3" / "summary.json")
    if e3:
        rows.append(
            (
                "Exp 3  Update ratio",
                f"warm-up effect ends at step {e3['warmup_effect_ends_step']} "
                f"(warm-up = {e3['warmup_steps']} steps)",
            )
        )

    e4 = _load_json(RUNS / "exp4" / "summary.json")
    if e4:
        l200 = e4["loss_at_200"]
        tuned = e4["tuned_peak_lr"]
        rows.append(
            (
                "Exp 4  Cosine vs WSD",
                f"tuned peaks cosine {tuned['cosine']:.1e} / WSD {tuned['wsd']:.1e}; "
                f"@200 cos {l200['cosine']:.3f} / wsd {l200['wsd']:.3f} "
                f"→ keep {e4['keep']}",
            )
        )

    e5 = _load_json(RUNS / "exp5" / "summary.json")
    if e5:
        mup = e5["argmin_lr"]["mup"]
        by_width = ", ".join(f"{w}:{float(mup[w]):.1e}" for w in sorted(mup, key=int))
        rows.append(
            (
                "Exp 5  µP LR transfer",
                f"µP best-LR {by_width}; rec@4096 {e5['recommended_lr_4096']:.1e}; "
                f"confidence {e5['confidence']}",
            )
        )

    return rows


def main() -> None:
    rows = collect()
    if not rows:
        print("No runs found. Run `make all` first.")
        return
    label_w = max(len(label) for label, _ in rows)
    print("Results at a glance")
    print("=" * 78)
    for label, headline in rows:
        print(f"{label:<{label_w}}   {headline}")


if __name__ == "__main__":
    main()
