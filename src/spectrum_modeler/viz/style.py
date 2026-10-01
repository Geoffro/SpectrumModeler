"""Shared chart styling: palette, ink colors, and matplotlib settings, in light and dark.

The categorical colors are a validated colorblind-safe palette. Assign them in
order (A = slot 1, B = slot 2, ...); never cycle past the end. The dark theme
uses the same hues, stepped for the dark surface.

Plotting code reads colors from ``C`` (``C.SURFACE``, ``C.SERIES[0]``, ...), which
follows the current theme. Light is the default; the app switches to dark.
"""

import contextlib
import dataclasses
import functools
from collections.abc import Callable, Iterator

import matplotlib as mpl
import numpy as np
from matplotlib.colors import Colormap, LinearSegmentedColormap
from matplotlib.figure import Figure


@dataclasses.dataclass(frozen=True)
class Theme:
    name: str
    SURFACE: str
    INK: str
    INK_SECONDARY: str
    MUTED: str
    GRID: str
    BASELINE: str
    HOVER: str  # button hover fill
    SERIES: tuple[str, ...]
    BLUES: tuple[str, ...]  # ordered series, least to most prominent
    LEVEL_CMAP: Colormap  # continuous level (dB): quiet recedes to the surface
    DIFFERENCE_CMAP: Colormap  # blue (A stronger), neutral midpoint, orange (B stronger)


# Blue ramp, light to dark (steps 250 to 600).
_RAMP = ("#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95")

LIGHT = Theme(
    name="light",
    SURFACE="#fcfcfb",
    INK="#0b0b0b",
    INK_SECONDARY="#52514e",
    MUTED="#898781",
    GRID="#e1e0d9",
    BASELINE="#c3c2b7",
    HOVER="#c3c2b7",
    SERIES=("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"),
    BLUES=_RAMP,
    LEVEL_CMAP=LinearSegmentedColormap.from_list(
        "level", ["#fcfcfb", *_RAMP, "#104281", "#0d366b"]
    ),
    DIFFERENCE_CMAP=LinearSegmentedColormap.from_list(
        "a_vs_b", ["#184f95", "#5598e7", "#f0efec", "#f19a73", "#b8481c"]
    ),
)

DARK = Theme(
    name="dark",
    SURFACE="#1a1a19",
    INK="#ffffff",
    INK_SECONDARY="#c3c2b7",
    MUTED="#898781",
    GRID="#383835",
    BASELINE="#52514e",
    HOVER="#52514e",
    SERIES=("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"),
    BLUES=_RAMP[::-1],
    LEVEL_CMAP=LinearSegmentedColormap.from_list(
        "level_dark", ["#1a1a19", "#0d366b", "#104281", *_RAMP[::-1], "#cde2fb"]
    ),
    DIFFERENCE_CMAP=LinearSegmentedColormap.from_list(
        "a_vs_b_dark", ["#86b6ef", "#3987e5", "#383835", "#d95926", "#f19a73"]
    ),
)

THEMES = {t.name: t for t in (LIGHT, DARK)}
_current = LIGHT
_pyplot = True  # new figures belong to pyplot (scripts, notebooks), not to the app


class _Current:
    """The current theme's colors, looked up when used."""

    def __getattr__(self, name: str):
        return getattr(_current, name)


C = _Current()


def current_theme() -> Theme:
    return _current


def set_theme(name: str) -> None:
    """Switch every later plot to the ``"light"`` or ``"dark"`` theme."""
    global _current
    _current = THEMES[name]


@contextlib.contextmanager
def using_theme(name: str) -> Iterator[None]:
    """Draw with another theme for a while (``using_theme("light")`` for saved files)."""
    previous = _current.name
    set_theme(name)
    try:
        with mpl.rc_context(rc()):
            yield
    finally:
        set_theme(previous)


def rc() -> dict:
    """Matplotlib settings for the current theme."""
    t = _current
    return {
        "figure.facecolor": t.SURFACE,
        "axes.facecolor": t.SURFACE,
        "savefig.facecolor": t.SURFACE,
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "text.color": t.INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlecolor": t.INK,
        "axes.labelcolor": t.INK_SECONDARY,
        "axes.edgecolor": t.BASELINE,
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "axes.prop_cycle": mpl.cycler(color=t.SERIES),
        "grid.color": t.GRID,
        "grid.linewidth": 0.8,
        "xtick.color": t.MUTED,
        "ytick.color": t.MUTED,
        "xtick.labelcolor": t.INK_SECONDARY,
        "ytick.labelcolor": t.INK_SECONDARY,
        "lines.linewidth": 2.0,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "legend.frameon": False,
        "legend.fontsize": 9,
        "legend.labelcolor": t.INK,
    }


def styled[F: Callable](func: F) -> F:
    """Run a plotting function with the project style applied."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with mpl.rc_context(rc()):
            return func(*args, **kwargs)

    return wrapper  # type: ignore[return-value]


def use_style() -> None:
    """Apply the project style globally.

    For interactive windows: text picks its font when drawn, which happens after
    ``styled`` has already restored the previous settings.
    """
    mpl.rcParams.update(rc())


def use_pyplot(on: bool) -> None:
    """Whether new figures are made through pyplot. The app turns this off: its
    figures are embedded or saved, and pyplot would open windows of its own."""
    global _pyplot
    _pyplot = on


def figure(fig: Figure | None, figsize: tuple[float, float]) -> Figure:
    """A new figure, or ``fig`` cleared for reuse (a figure embedded in the app)."""
    if fig is None:
        if not _pyplot:
            return Figure(figsize=figsize, layout="constrained")
        import matplotlib.pyplot as plt

        return plt.figure(figsize=figsize, layout="constrained")
    fig.clear()
    fig.set_facecolor(_current.SURFACE)
    fig.set_layout_engine("constrained")
    return fig


# --- Widget colors (explorer and tuner windows) ------------------------------------------


def radio_props() -> dict:
    return {"edgecolor": _current.INK_SECONDARY}


def check_props() -> dict:
    """Keyword arguments for ``CheckButtons``."""
    return {
        "frame_props": {"edgecolor": _current.INK_SECONDARY},
        "check_props": {"color": _current.SERIES[0], "linewidth": 2},
    }


def slider_props() -> dict:
    """Keyword arguments for ``Slider`` (no marker at the initial value)."""
    return {
        "color": _current.SERIES[0],
        "track_color": _current.GRID,
        "initcolor": "none",
        "handle_style": {"facecolor": _current.INK, "edgecolor": _current.SURFACE},
    }


def button_props(ax) -> dict:
    """Keyword arguments for ``Button``; also drops ``ax``'s frame (the fill is enough)."""
    for spine in ax.spines.values():
        spine.set_visible(False)
    return {"color": _current.GRID, "hovercolor": _current.HOVER}


def ordinal_blues(k: int) -> list[str]:
    """``k`` evenly spaced steps of the blue ramp, least to most prominent."""
    blues = _current.BLUES
    if k > len(blues):
        raise ValueError(f"At most {len(blues)} ordered series; got {k}")
    idx = np.round(np.linspace(0, len(blues) - 1, k)).astype(int)
    return [blues[i] for i in idx]
