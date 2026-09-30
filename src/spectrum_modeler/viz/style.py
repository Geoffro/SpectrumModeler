"""Shared chart styling: palette, ink colors, and matplotlib settings.

The categorical colors are a validated colorblind-safe palette. Assign them in
order (A = slot 1, B = slot 2, ...); never cycle past the end.
"""

import functools
from collections.abc import Callable

import matplotlib as mpl
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")

# Single-hue ramp for ordered series (mode number, partial-sum count), light to dark.
BLUES = ("#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95")

# Continuous level (dB) scale: quiet recedes to the surface, loud is dark blue.
LEVEL_CMAP = LinearSegmentedColormap.from_list("level", [SURFACE, *BLUES, "#104281", "#0d366b"])

RC = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "text.color": INK,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.titlecolor": INK,
    "axes.labelcolor": INK_SECONDARY,
    "axes.edgecolor": BASELINE,
    "axes.linewidth": 1.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "axes.axisbelow": True,
    "axes.prop_cycle": mpl.cycler(color=SERIES),
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK_SECONDARY,
    "ytick.labelcolor": INK_SECONDARY,
    "lines.linewidth": 2.0,
    "lines.solid_capstyle": "round",
    "lines.solid_joinstyle": "round",
    "legend.frameon": False,
    "legend.fontsize": 9,
}


def styled[F: Callable](func: F) -> F:
    """Run a plotting function with the project style applied."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with mpl.rc_context(RC):
            return func(*args, **kwargs)

    return wrapper  # type: ignore[return-value]


def use_style() -> None:
    """Apply the project style globally.

    For interactive windows: text picks its font when drawn, which happens after
    ``styled`` has already restored the previous settings.
    """
    mpl.rcParams.update(RC)


def ordinal_blues(k: int) -> list[str]:
    """``k`` evenly spaced steps of the blue ramp, light to dark."""
    if k > len(BLUES):
        raise ValueError(f"At most {len(BLUES)} ordered series; got {k}")
    idx = np.round(np.linspace(0, len(BLUES) - 1, k)).astype(int)
    return [BLUES[i] for i in idx]
