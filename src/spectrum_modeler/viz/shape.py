"""String-shape plots: the plucked triangle and how it is built from modes."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ..modal import initial_shape, mode_contributions
from ..params import PluckParams, StringParams
from .style import BASELINE, INK, INK_SECONDARY, MUTED, SERIES, SURFACE, ordinal_blues, styled


def _grid(string: StringParams, pluck: PluckParams, points: int = 801) -> np.ndarray:
    """Positions along the string in m, including the pluck point exactly."""
    xp = pluck.beta(string.length) * string.length
    return np.union1d(np.linspace(0, string.length, points), [xp])


def frame_string_axes(ax: Axes, string: StringParams, pluck: PluckParams) -> None:
    """Axes limits, labels, baseline, and saddle/nut marks for a string-shape plot."""
    h_mm = pluck.amplitude * 1000
    ax.axhline(0, color=BASELINE, linewidth=1, zorder=1)
    ax.set_xlim(0, string.length * 1000)
    ax.set_ylim(-0.3 * h_mm, 1.3 * h_mm)
    ax.set_xlabel("Distance from saddle (mm)")
    ax.set_ylabel("Displacement (mm)")
    for x, label, ha in ((0, "saddle", "left"), (string.length * 1000, "nut", "right")):
        ax.annotate(
            label,
            (x, 0),
            xytext=(0, -12),
            textcoords="offset points",
            ha=ha,
            va="top",
            color=MUTED,
            fontsize=8,
        )


@styled
def plot_initial_shape(
    string: StringParams,
    pluck: PluckParams,
    ax: Axes | None = None,
    *,
    color: str = SERIES[0],
    title: str | None = None,
) -> Axes:
    """The triangular shape at release, with the pluck point marked."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 2.6), layout="constrained")
    x = _grid(string, pluck)
    xp_mm = pluck.beta(string.length) * string.length * 1000
    h_mm = pluck.amplitude * 1000

    ax.plot(x * 1000, initial_shape(x, string.length, pluck) * 1000, color=color, zorder=3)
    ax.plot(xp_mm, h_mm, "o", markersize=8, color=color, mec=SURFACE, mew=2, zorder=4)
    ax.annotate(
        f"pluck, {pluck.distance_mm:g} mm",
        (xp_mm, h_mm),
        xytext=(8, 4),
        textcoords="offset points",
        color=INK_SECONDARY,
        fontsize=9,
    )
    frame_string_axes(ax, string, pluck)
    ax.set_title(title if title is not None else f"{string.note} string at release")
    return ax


@styled
def plot_mode_overlay(
    string: StringParams, pluck: PluckParams, n_modes: int = 4, ax: Axes | None = None
) -> Axes:
    """The first few mode shapes, each scaled by its amplitude, over the initial shape."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3), layout="constrained")
    x = _grid(string, pluck)
    modes = mode_contributions(x, string, pluck, n_modes) * 1000

    ax.plot(
        x * 1000, initial_shape(x, string.length, pluck) * 1000, color=INK, label="initial shape"
    )
    for i, (mode, color) in enumerate(zip(modes, ordinal_blues(n_modes), strict=True)):
        ax.plot(x * 1000, mode, color=color, linewidth=1.2, label=f"mode {i + 1}")
    frame_string_axes(ax, string, pluck)
    h_mm = pluck.amplitude * 1000
    ax.set_ylim(min(-0.3 * h_mm, 1.15 * modes.min()), 1.3 * h_mm)
    ax.set_title(f"First {n_modes} modes")
    ax.legend(loc="upper right", ncols=2)
    return ax


@styled
def plot_partial_sums(
    string: StringParams,
    pluck: PluckParams,
    counts: tuple[int, ...] = (1, 3, 10, 40),
    ax: Axes | None = None,
) -> Axes:
    """Sums of the first N modes, converging on the triangle."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3), layout="constrained")
    x = _grid(string, pluck)
    cumulative = np.cumsum(mode_contributions(x, string, pluck, max(counts)), axis=0) * 1000

    ax.plot(
        x * 1000,
        initial_shape(x, string.length, pluck) * 1000,
        color=MUTED,
        linewidth=1,
        label="initial shape",
    )
    for count, color in zip(counts, ordinal_blues(len(counts)), strict=True):
        label = f"{count} mode" + ("s" if count > 1 else "")
        ax.plot(x * 1000, cumulative[count - 1], color=color, linewidth=1.5, label=label)
    frame_string_axes(ax, string, pluck)
    ax.set_title("Adding modes rebuilds the shape")
    ax.legend(loc="upper right")
    return ax


@styled
def plot_mode_decomposition(string: StringParams, pluck: PluckParams) -> Figure:
    """Mode overlay and partial sums side by side."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 3.6), layout="constrained")
    plot_mode_overlay(string, pluck, ax=left)
    plot_partial_sums(string, pluck, ax=right)
    beta = pluck.beta(string.length)
    fig.suptitle(
        f"{string.note}, plucked {pluck.distance_mm:g} mm from the saddle (β = {beta:.3f})",
        x=0.01,
        ha="left",
        fontweight="bold",
    )
    return fig
