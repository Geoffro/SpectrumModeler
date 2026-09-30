"""Harmonic spectrum plots: bar charts in dB, and A/B comparison of two plucks."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ..analysis import spectral_centroid
from ..modal import ModalSpectrum, coefficients, envelope_nulls
from ..params import ModelOptions, StringParams
from ..presets import pluck
from .shape import plot_initial_shape
from .style import MUTED, SERIES, styled

OBSERVABLE_LABELS = {
    "displacement": "string displacement",
    "velocity": "string velocity",
    "bridge_force": "bridge force",
}


def node_markers(ax: Axes, nulls: np.ndarray):
    """Small triangles along the top edge where the pluck point is a node."""
    (line,) = ax.plot(
        nulls,
        np.full(nulls.size, 0.985),
        "v",
        markersize=5,
        color=MUTED,
        transform=ax.get_xaxis_transform(),
        clip_on=False,
        zorder=4,
        label="node at pluck point",
    )
    return line


def describe(spectrum: ModalSpectrum) -> str:
    text = (
        f"{spectrum.string.note}, {spectrum.pluck.distance_mm:g} mm from saddle "
        f"(β = {spectrum.beta:.3f})"
    )
    if spectrum.pluck.width_mm > 0:
        text += f", {spectrum.pluck.width_mm:.3g} mm finger"
    return text


@styled
def plot_spectrum(
    spectrum: ModalSpectrum,
    ax: Axes | None = None,
    *,
    ref: float | None = None,
    floor_db: float = -60.0,
    color: str = SERIES[0],
    title: str | None = None,
) -> Axes:
    """Harmonic levels in dB, with markers where the pluck point is a node.

    ``ref`` sets 0 dB (default: this spectrum's strongest harmonic). Pass a shared
    ``ref`` to compare spectra on one scale.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3.5), layout="constrained")
    n = spectrum.harmonics
    levels = spectrum.levels_db(ref=ref, floor_db=floor_db)

    ax.bar(n, levels - floor_db, bottom=floor_db, width=0.6, color=color, linewidth=0, zorder=3)

    nulls = envelope_nulls(spectrum.beta, n[-1] + 0.5)
    if nulls.size:
        node_markers(ax, nulls)
        ax.legend(loc="upper right", fontsize=8, handletextpad=0.2)

    ax.set_xlim(0.3, n[-1] + 0.7)
    ax.set_ylim(floor_db, 5)
    ticks = [1, *range(5, n[-1] + 1, 5)]
    ax.set_xticks(ticks)
    ax.set_xlabel("Harmonic number")
    ax.set_ylabel("Level (dB)")
    ax.set_title(title if title is not None else describe(spectrum))
    return ax


@styled
def plot_comparison(
    a: ModalSpectrum,
    b: ModalSpectrum,
    *,
    labels: tuple[str, str] | None = None,
    floor_db: float = -60.0,
) -> Figure:
    """Two plucks side by side: string shape on top, spectrum below, on one dB scale."""
    labels = labels or (describe(a), describe(b))
    ref = max(np.abs(a.amps).max(), np.abs(b.amps).max())

    fig, axes = plt.subplots(
        2, 2, figsize=(13, 6.5), height_ratios=[1, 1.7], sharey="row", layout="constrained"
    )
    for col, (spectrum, label, color) in enumerate(zip((a, b), labels, SERIES, strict=False)):
        plot_initial_shape(
            spectrum.string, spectrum.pluck, ax=axes[0, col], color=color, title=label
        )
        plot_spectrum(spectrum, ax=axes[1, col], ref=ref, floor_db=floor_db, color=color, title="")
    axes[1, 0].set_ylabel("Level (dB re strongest harmonic)")
    axes[1, 1].set_ylabel("")
    axes[0, 1].set_ylabel("")

    fig.suptitle(
        f"{a.string.note}: {OBSERVABLE_LABELS[a.observable]} spectrum, same pluck height",
        x=0.01,
        ha="left",
        fontweight="bold",
        fontsize=13,
    )
    return fig


@styled
def plot_brightness_vs_position(
    string: StringParams,
    positions_mm: np.ndarray,
    *,
    fingers: tuple[str, ...] = ("ideal", "nail", "flesh"),
    opts: ModelOptions | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Spectral centroid against pluck position, one line per finger type.

    The centroid is divided by f0, so it reads as "the average harmonic". In the
    uncalibrated model this curve is the same for every open string: the
    spectrum in harmonic numbers depends only on beta (and the finger).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4.5), layout="constrained")
    positions_mm = np.asarray(positions_mm, dtype=float)
    for finger, color in zip(fingers, SERIES, strict=False):
        centroids = []
        for mm in positions_mm:
            spectrum = coefficients(string, pluck(mm, length=string.length, finger=finger), opts)
            centroids.append(spectral_centroid(spectrum.freqs, spectrum.amps) / spectrum.string.f0)
        ax.plot(positions_mm, centroids, color=color, label=finger)
    ax.set_xlim(positions_mm[0], positions_mm[-1])
    ax.set_xlabel("Pluck distance from saddle (mm)")
    ax.set_ylabel("Spectral centroid (× f0)")
    ax.set_title(f"{string.note}: brightness falls as the pluck moves away from the bridge")
    ax.legend(title="finger", loc="upper right")
    return ax
