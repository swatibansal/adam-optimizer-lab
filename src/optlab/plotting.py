"""All matplotlib lives here — one function per figure.

House style (enforced by convention, see CLAUDE.md):

* 150 dpi PNG.
* Title + axis labels on every axes.
* A one-sentence plain-English caption rendered at the bottom via ``fig.text``.
* Log-scale x-axis for any learning-rate axis.

Every function takes already-computed numbers and an output path; no training or
math happens here.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless / deterministic
import matplotlib.pyplot as plt  # noqa: E402

DPI = 150


def _caption(fig: plt.Figure, text: str) -> None:
    """Render a one-sentence caption at the bottom of the figure."""
    fig.text(0.5, 0.01, text, ha="center", va="bottom", fontsize=9, wrap=True, style="italic")


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Leave room at the bottom for the caption.
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def plot_exp2(
    steps: Sequence[int],
    step_size_corrected: Sequence[float],
    step_size_uncorrected: Sequence[float],
    weight_corrected: Sequence[float],
    weight_uncorrected: Sequence[float],
    path: Path,
    caption: str,
) -> None:
    """Two panels: (a) step size per step, (b) weight trajectory — correction on/off."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.plot(steps, step_size_corrected, "-o", ms=3, label="correction on")
    ax1.plot(steps, step_size_uncorrected, "-s", ms=3, label="correction off")
    ax1.set_title("(a) Step size per step")
    ax1.set_xlabel("step")
    ax1.set_ylabel("size of weight move")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(steps, weight_corrected, "-o", ms=3, label="correction on")
    ax2.plot(steps, weight_uncorrected, "-s", ms=3, label="correction off")
    ax2.set_title("(b) Weight trajectory")
    ax2.set_xlabel("step")
    ax2.set_ylabel("weight value")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    _caption(fig, caption)
    _save(fig, path)


def plot_exp3(
    steps: Sequence[int],
    ratios_by_group: dict[str, Sequence[float]],
    warmup_steps: int,
    stabilized_step: int | None,
    path: Path,
    caption: str,
) -> None:
    """Log-scale update-ratio per group, warm-up shaded, stabilization marked."""
    fig, ax = plt.subplots(figsize=(10, 5.5))

    for name, ratios in ratios_by_group.items():
        ax.plot(steps, ratios, label=name, lw=1.3)

    ax.axvspan(steps[0], warmup_steps, color="gray", alpha=0.15, label="warm-up window")
    if stabilized_step is not None:
        ax.axvline(
            stabilized_step,
            color="black",
            ls="--",
            lw=1.2,
            label=f"ratios stabilized (step {stabilized_step})",
        )
    ax.axhline(1e-3, color="green", ls=":", lw=1.0, label="rule-of-thumb 1e-3")

    ax.set_yscale("log")
    ax.set_title("Per-layer update ratio  ‖Δw‖ / ‖w‖  during training")
    ax.set_xlabel("step")
    ax.set_ylabel("update ratio (log scale)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, which="both", alpha=0.3)

    _caption(fig, caption)
    _save(fig, path)


def plot_exp4(
    steps_stop: Sequence[int],
    lr_cosine: Sequence[float],
    lr_wsd: Sequence[float],
    loss_cosine_stop: Sequence[float],
    loss_wsd_stop: Sequence[float],
    steps_full: Sequence[int],
    loss_cosine_full: Sequence[float],
    loss_wsd_full: Sequence[float],
    stop_at: int,
    path: Path,
    caption: str,
) -> None:
    """Two panels: (a) LR schedules, (b) loss curves with a marker at ``stop_at``."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(steps_full, lr_cosine, label="cosine", color="C0")
    ax1.plot(steps_full, lr_wsd, label="WSD", color="C1")
    ax1.axvline(stop_at, color="black", ls="--", lw=1.0)
    ax1.set_title("(a) Learning-rate schedules")
    ax1.set_ylabel("learning rate")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Full 300-step runs as dashed lines, the stopped-at-200 runs as solid.
    ax2.plot(steps_full, loss_cosine_full, "--", color="C0", alpha=0.6, label="cosine (full 300)")
    ax2.plot(steps_full, loss_wsd_full, "--", color="C1", alpha=0.6, label="WSD (full 300)")
    ax2.plot(steps_stop, loss_cosine_stop, "-", color="C0", label="cosine (stopped 200)")
    ax2.plot(steps_stop, loss_wsd_stop, "-", color="C1", label="WSD (stopped 200)")
    ax2.axvline(stop_at, color="black", ls="--", lw=1.0)
    ax2.set_title("(b) Training loss")
    ax2.set_xlabel("step")
    ax2.set_ylabel("loss")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(True, alpha=0.3)

    # The whole story lives in a narrow band near the end, invisible on the full-range
    # axis above — so zoom into late training in an inset where the schedule difference
    # (and the WSD final drop) is actually legible.
    zoom_start = 130
    axins = ax2.inset_axes([0.52, 0.42, 0.45, 0.5])
    axins.plot(steps_full, loss_cosine_full, "--", color="C0", alpha=0.7)
    axins.plot(steps_full, loss_wsd_full, "--", color="C1", alpha=0.7)
    axins.plot(steps_stop, loss_cosine_stop, "-", color="C0")
    axins.plot(steps_stop, loss_wsd_stop, "-", color="C1")
    axins.axvline(stop_at, color="black", ls="--", lw=0.8)
    _annotate_point(axins, steps_stop, loss_cosine_stop, stop_at, "C0")
    _annotate_point(axins, steps_stop, loss_wsd_stop, stop_at, "C1")
    _annotate_point(axins, steps_full, loss_cosine_full, steps_full[-1], "C0")
    _annotate_point(axins, steps_full, loss_wsd_full, steps_full[-1], "C1")

    tail = [
        v
        for series in (loss_cosine_full, loss_wsd_full)
        for s, v in zip(steps_full, series, strict=True)
        if s >= zoom_start
    ]
    axins.set_xlim(zoom_start, steps_full[-1])
    axins.set_ylim(min(tail) * 0.95, max(tail) * 1.08)
    axins.set_title("zoom: late training", fontsize=8)
    axins.tick_params(labelsize=7)
    axins.grid(True, alpha=0.3)
    ax2.indicate_inset_zoom(axins, edgecolor="gray")

    _caption(fig, caption)
    _save(fig, path)


def _annotate_point(
    ax: plt.Axes,
    steps: Sequence[int],
    values: Sequence[float],
    at_step: int,
    color: str,
) -> None:
    if at_step not in steps:
        return
    idx = list(steps).index(at_step)
    y = values[idx]
    ax.annotate(
        f"{y:.3f}",
        xy=(at_step, y),
        xytext=(4, 4),
        textcoords="offset points",
        fontsize=8,
        color=color,
    )


def plot_exp5(
    lrs: Sequence[float],
    losses_std: dict[int, Sequence[float]],
    losses_mup: dict[int, Sequence[float]],
    argmin_std: dict[int, float],
    argmin_mup: dict[int, float],
    path: Path,
    caption: str,
) -> None:
    """Two panels (standard vs µP): loss-vs-lr per width, each minimum marked."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    _lr_sweep_panel(ax1, lrs, losses_std, argmin_std, "Standard parameterization")
    _lr_sweep_panel(ax2, lrs, losses_mup, argmin_mup, "µP parameterization")

    _caption(fig, caption)
    _save(fig, path)


def _lr_sweep_panel(
    ax: plt.Axes,
    lrs: Sequence[float],
    losses_by_width: dict[int, Sequence[float]],
    argmin_by_width: dict[int, float],
    title: str,
) -> None:
    for i, (width, losses) in enumerate(sorted(losses_by_width.items())):
        color = f"C{i}"
        ax.plot(lrs, losses, "-o", ms=3, color=color, label=f"width {width}")
        best_lr = argmin_by_width[width]
        best_idx = list(lrs).index(best_lr)
        ax.plot(best_lr, losses[best_idx], "o", ms=10, mfc="none", mec=color, mew=2)
        ax.annotate(
            f"{best_lr:.1e}",
            xy=(best_lr, losses[best_idx]),
            xytext=(0, -14),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=color,
        )
    ax.set_xscale("log")
    # Log y as well: the loss blows up at the largest learning rates, which would
    # otherwise flatten the informative U near each curve's minimum.
    ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel("learning rate (log scale)")
    ax.set_ylabel("final loss (log scale, mean of last 10 steps)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
