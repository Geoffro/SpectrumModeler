"""Two recordings compared: spectrograms side by side, their difference, and per-harmonic views.

A is blue and B is orange throughout, as elsewhere; the difference map uses the
same pair as a diverging scale with a neutral grey midpoint.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from ..compare import FLOOR_DB, Comparison, harmonic_track
from .spectrogram import plot_spectrogram
from .style import C, styled

DIFFERENCE_RANGE_DB = 20.0


@styled
def plot_comparison(
    comparison: Comparison,
    *,
    max_freq: float = 5000.0,
    harmonics: tuple[int, ...] = (1, 2, 3, 4),
    level_harmonics: int = 30,
) -> Figure:
    a, b = comparison.a, comparison.b
    fig, axes = plt.subplot_mosaic(
        [
            ["a", "a", "b", "b"],
            ["d", "d", "d", "d"],
            ["h", "h", "k1", "k2"],
            ["h", "h", "k3", "k4"],
        ],
        figsize=(14, 13),
        height_ratios=[1.1, 1.1, 0.75, 0.75],
        layout="constrained",
    )

    # --- Side by side on one scale ------------------------------------------------------
    for key, rec, colorbar in (("a", a, False), ("b", b, True)):
        plot_spectrogram(
            rec.spectrogram, axes[key], ref=comparison.ref, floor_db=FLOOR_DB,
            max_freq=max_freq, title=f"{'AB'[key == 'b']}: {rec.name}", colorbar=colorbar,
        )  # fmt: skip
    axes["b"].set_ylabel("")

    # --- Difference -------------------------------------------------------------------------
    ax = axes["d"]
    keep = a.spectrogram.freqs <= max_freq
    mesh = ax.pcolormesh(
        a.spectrogram.times,
        a.spectrogram.freqs[keep],
        comparison.difference_db()[keep],
        cmap=C.DIFFERENCE_CMAP,
        vmin=-DIFFERENCE_RANGE_DB,
        vmax=DIFFERENCE_RANGE_DB,
        shading="nearest",
        rasterized=True,
    )
    ax.set_facecolor(C.SURFACE)
    ax.grid(False)
    ax.set_xlabel("Time after the pluck (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("Difference, B − A (blank where both are near the noise floor)")
    bar = fig.colorbar(mesh, ax=ax, pad=0.01, extend="both")
    bar.set_label("dB   (+ B stronger, − A stronger)")

    # --- Harmonic levels at the pluck --------------------------------------------------------
    ax = axes["h"]
    measured = [r.measurement for r in (a, b) if r.measurement is not None]
    if measured:
        top = max(np.nanmax(m.initial_db) for m in measured)
        for rec, color, style in ((a, C.SERIES[0], "bar"), (b, C.SERIES[1], "dot")):
            m = rec.measurement
            if m is None:
                continue
            ok = m.reliable & np.isfinite(m.initial_db) & (m.harmonics <= level_harmonics)
            levels = m.initial_db[ok] - top
            if style == "bar":
                ax.bar(m.harmonics[ok], levels + 60, bottom=-60, width=0.6, color=color,
                       linewidth=0, label=f"A: {rec.name}", zorder=3)  # fmt: skip
            else:
                ax.plot(m.harmonics[ok], levels, "o", markersize=6, color=color, mec=C.SURFACE,
                        mew=1, label=f"B: {rec.name}", zorder=4)  # fmt: skip
        ax.set_xlim(0.3, level_harmonics + 0.7)
        ax.set_ylim(-60, 5)
        ax.set_xticks([1, *range(5, level_harmonics + 1, 5)])
        ax.legend(loc="upper right", fontsize=8)
    _note_problems(ax, comparison)
    ax.set_xlabel("Harmonic number")
    ax.set_ylabel("Level at the pluck (dB re strongest)")
    ax.set_title("Harmonic levels, extrapolated to the moment of the pluck")

    # --- Decay of individual harmonics -------------------------------------------------------
    times = a.spectrogram.times
    for key, harmonic in zip(("k1", "k2", "k3", "k4"), harmonics, strict=False):
        ax = axes[key]
        for rec, color in ((a, C.SERIES[0]), (b, C.SERIES[1])):
            track = harmonic_track(comparison, rec, harmonic)
            if track is not None:
                ax.plot(times, track, color=color, linewidth=1.5)
        ax.set_ylim(FLOOR_DB, 0)
        ax.set_xlim(times[0], times[-1])
        ax.set_title(f"Harmonic {harmonic}", fontsize=9)
        ax.tick_params(labelsize=8)
        if key in ("k3", "k4"):
            ax.set_xlabel("Time (s)", fontsize=9)
        if key in ("k1", "k3"):
            ax.set_ylabel("dB", fontsize=9)
    for key in ("k1", "k2", "k3", "k4")[len(harmonics) :]:
        axes[key].set_visible(False)

    fig.suptitle(
        f"A: {a.name}  vs  B: {b.name} · first {comparison.duration:.1f} s after each pluck, "
        "matched in loudness",
        x=0.01, ha="left", fontweight="bold", fontsize=13,
    )  # fmt: skip
    return fig


def _note_problems(ax, comparison: Comparison) -> None:
    recordings = (comparison.a, comparison.b)
    problems = [
        f"{'AB'[i]} not measured: {r.problem}" for i, r in enumerate(recordings) if r.problem
    ]
    if problems:
        ax.text(
            0.01,
            0.02,
            "\n".join(problems),
            transform=ax.transAxes,
            fontsize=8,
            color=C.MUTED,
            va="bottom",
            bbox={"facecolor": C.SURFACE, "edgecolor": C.GRID},
        )
