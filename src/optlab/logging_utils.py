"""CSV / JSON run logging under ``runs/``.

Kept deliberately tiny: a helper to append typed rows to a CSV and a helper to
write a ``summary.json``. Numbers are formatted with ``repr`` for floats so that
two runs with the same seed produce byte-identical CSV files.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

# All logs live under <repo>/runs/. Resolve relative to this file so scripts work
# regardless of the current working directory.
RUNS_DIR = Path(__file__).resolve().parents[2] / "runs"


def run_path(experiment: str, filename: str) -> Path:
    """Return ``runs/<experiment>/<filename>`` and ensure the directory exists."""
    directory = RUNS_DIR / experiment
    directory.mkdir(parents=True, exist_ok=True)
    return directory / filename


class CsvLogger:
    """Append rows to a CSV with a fixed set of columns.

    Usage::

        log = CsvLogger(run_path("exp3", "update_ratio.csv"), ["step", "group", "ratio", "lr"])
        log.write({"step": 1, "group": "embed", "ratio": 0.001, "lr": 3e-3})
        log.close()
    """

    def __init__(self, path: Path, columns: list[str]) -> None:
        self.path = path
        self.columns = columns
        self._fh = open(path, "w", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=columns)
        self._writer.writeheader()

    def write(self, row: dict[str, Any]) -> None:
        self._writer.writerow({c: _fmt(row[c]) for c in self.columns})
        # Flush each row so a long run's progress is visible on disk immediately.
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> CsvLogger:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def write_summary(experiment: str, summary: dict[str, Any]) -> Path:
    """Write ``runs/<experiment>/summary.json`` (sorted keys, 2-space indent)."""
    path = run_path(experiment, "summary.json")
    with open(path, "w") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return path


def _fmt(value: Any) -> Any:
    """Format floats stably so identical seeds give identical bytes."""
    if isinstance(value, float):
        return repr(value)
    return value
