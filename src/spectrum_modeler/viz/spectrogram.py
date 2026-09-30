"""Spectrogram plots: how each harmonic's level changes over time."""

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ..analysis import Spectrogram
from .style import LEVEL_CMAP, styled


@styled
def plot_spectrogram(
    spec: Spectrogram,
    ax: Axes | None = None,
    *,
    ref: float | None = None,
    floor_db: float = -80.0,
    max_freq: float = 5000.0,
    title: str = "",
    colorbar: bool = True,
) -> Axes:
    """Level in dB against time and frequency, up to ``max_freq`` Hz."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    keep = spec.freqs <= max_freq
    mesh = ax.pcolormesh(
        spec.times,
        spec.freqs[keep],
        spec.db(ref=ref, floor_db=floor_db)[keep],
        cmap=LEVEL_CMAP,
        vmin=floor_db,
        vmax=0,
        shading="nearest",
        rasterized=True,
    )
    ax.grid(False)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)
    if colorbar:
        ax.figure.colorbar(mesh, ax=ax, label="Level (dB)", pad=0.01)
    return ax


@styled
def plot_spectrogram_pair(
    a: Spectrogram,
    b: Spectrogram,
    labels: tuple[str, str],
    *,
    suptitle: str = "",
    floor_db: float = -80.0,
    max_freq: float = 5000.0,
) -> Figure:
    """Two spectrograms side by side on one dB scale."""
    ref = max(a.magnitude.max(), b.magnitude.max())
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True, layout="constrained")
    plot_spectrogram(
        a, left, ref=ref, floor_db=floor_db, max_freq=max_freq, title=labels[0], colorbar=False
    )
    plot_spectrogram(b, right, ref=ref, floor_db=floor_db, max_freq=max_freq, title=labels[1])
    right.set_ylabel("")
    if suptitle:
        fig.suptitle(suptitle, x=0.01, ha="left", fontweight="bold", fontsize=13)
    return fig
