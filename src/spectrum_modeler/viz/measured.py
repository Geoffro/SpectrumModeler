"""Plots of a measured pluck: its spectrum, harmonic levels against the model, and decay."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from ..analysis import spectrogram
from ..measure import Measurement, align_model_db, window_spectrum
from ..modal import ModalSpectrum
from .spectrogram import plot_spectrogram
from .style import C, styled


def describe(m: Measurement) -> str:
    parts = [m.clip or "recording"]
    if m.take is not None:
        parts[0] += f", take {m.take}"
    parts.append(f"f0 {m.f0:.2f} Hz")
    if m.inharmonicity_b is not None:
        parts.append(f"B {m.inharmonicity_b:.1e}")
    if m.sigma0 is not None:
        parts.append(f"σ0 {m.sigma0:.2f}/s, σ2 {m.sigma2:.1e}")
    parts.append(f"{m.n_reliable} of {m.harmonics.size} harmonics measured")
    return " · ".join(parts)


@styled
def plot_measurement(
    audio: np.ndarray,
    sample_rate: int,
    onset: int,
    m: Measurement,
    model: ModalSpectrum | None = None,
    *,
    floor_db: float = -60.0,
) -> Figure:
    """Three panels: the analysis-window spectrum, harmonic levels, and a spectrogram."""
    ws = window_spectrum(audio, sample_rate, onset, *m.window)
    top = min(float(m.freqs[-1]) * 1.03, sample_rate / 2)
    rel = m.reliable

    fig, (spec_ax, level_ax, time_ax) = plt.subplots(
        3, 1, figsize=(12, 11), height_ratios=[1.2, 1, 1], layout="constrained"
    )

    # Spectrum with detected harmonics
    keep = ws.freqs <= top
    spec_ax.plot(ws.freqs[keep], ws.db[keep], color=C.SERIES[0], linewidth=1, label="spectrum")
    spec_ax.plot(m.freqs, m.floor_db, color=C.MUTED, linewidth=1, label="local noise floor")
    spec_ax.plot(
        m.freqs[rel], m.window_db[rel], "o", markersize=6,
        color=C.SERIES[0], mec=C.SURFACE, mew=1.5, label="harmonic",
    )  # fmt: skip
    spec_ax.plot(
        m.freqs[~rel], m.window_db[~rel], "o", markersize=6,
        mfc="none", mec=C.MUTED, label="harmonic not found",
    )  # fmt: skip
    for i, f in enumerate(m.off_series_hz):
        spec_ax.axvline(
            f, color=C.SERIES[1], linewidth=1.2, label="off-series peak" if i == 0 else None
        )
    visible = ws.db[keep]
    spec_ax.set_ylim(np.percentile(visible, 1) - 5, visible.max() + 5)
    spec_ax.set_xlim(0, top)
    start, length = m.window
    spec_ax.set_title(
        f"Spectrum {start * 1000:.0f}–{(start + length) * 1000:.0f} ms after the pluck"
    )
    spec_ax.set_xlabel("Frequency (Hz)")
    spec_ax.set_ylabel("Level (dBFS)")
    spec_ax.legend(loc="upper right", ncols=5)

    # Harmonic levels at the pluck, against the model
    ref = np.nanmax(m.initial_db)
    measured = m.initial_db - ref
    level_ax.bar(
        m.harmonics[rel],
        measured[rel] - floor_db,
        bottom=floor_db,
        width=0.6,
        color=C.SERIES[0],
        linewidth=0,
        label="measured",
        zorder=3,
    )
    if model is not None:
        aligned = align_model_db(m, model) - ref
        level_ax.plot(
            m.harmonics, np.maximum(aligned, floor_db), "o", markersize=5,
            color=C.SERIES[1], mec=C.SURFACE, mew=1, label="model (uncalibrated)", zorder=4,
        )  # fmt: skip
    level_ax.set_ylim(floor_db, 5)
    level_ax.set_xlim(0.3, m.harmonics[-1] + 0.7)
    level_ax.set_xlabel("Harmonic number")
    level_ax.set_ylabel("Level at the pluck (dB re strongest)")
    level_ax.set_title("Harmonic levels, extrapolated back to the moment of the pluck")
    if model is not None:
        level_ax.legend(loc="upper right")
    else:
        level_ax.text(
            1, 1.02, "no model: note or pluck position unknown",
            transform=level_ax.transAxes, ha="right", va="bottom", color=C.INK_SECONDARY,
            fontsize=8,
        )  # fmt: skip

    # Decay over time
    plot_spectrogram(
        spectrogram(audio[onset:], sample_rate), time_ax, max_freq=top, title="Spectrogram"
    )

    fig.suptitle(describe(m), x=0.01, ha="left", fontweight="bold", fontsize=12)
    return fig
