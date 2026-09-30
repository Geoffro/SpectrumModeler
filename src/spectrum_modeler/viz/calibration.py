"""Calibration report: the fitted response curve, per-take errors, and pluck-position check."""

from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from ..calibrate import MAX_BETA_SHIFT_MM, CalibrationResult
from .style import GRID, INK_SECONDARY, MUTED, SERIES, SURFACE, styled

GOOD_FIT_DB = 3.0  # roadmap target: RMS error per take


@styled
def plot_calibration_report(
    result: CalibrationResult,
    true_response_db: Callable[[np.ndarray], np.ndarray] | None = None,
) -> Figure:
    """Three panels: H(f) with the data behind it, RMS error per take, fitted vs manifest position.

    ``true_response_db`` (synthetic sessions only) adds the known H(f) for comparison,
    shifted to the fitted curve's level since H is only defined up to a constant.
    """
    profile = result.profile
    fig, (h_ax, err_ax, pos_ax) = plt.subplots(
        3, 1, figsize=(12, 12), height_ratios=[1.3, 1, 1], layout="constrained"
    )

    # --- Response curve --------------------------------------------------------------
    freqs = np.array(profile.response_freqs)
    h_ax.scatter(
        result.freqs, result.response_data_db, s=8, color=MUTED, alpha=0.35, linewidths=0,
        label="measured harmonics (after removing the pluck)",
    )  # fmt: skip
    h_ax.plot(freqs, profile.response_db, color=SERIES[0], label="fitted H(f)", zorder=4)
    if true_response_db is not None:
        truth = true_response_db(freqs)
        truth += np.median(np.array(profile.response_db) - truth)
        h_ax.plot(freqs, truth, color=SERIES[1], linewidth=1.5, label="true H(f)", zorder=3)
    h_ax.set_xscale("log")
    h_ax.set_xlim(freqs[0], freqs[-1])
    spread = np.percentile(result.response_data_db, [1, 99])
    h_ax.set_ylim(spread[0] - 5, spread[1] + 5)
    h_ax.set_xlabel("Frequency (Hz)")
    h_ax.set_ylabel("Gain (dB, relative)")
    h_ax.set_title("Body, room, and microphone response")
    h_ax.legend(loc="lower left")
    h_ax.grid(True, axis="x", which="major", color=GRID)

    # --- Error per take ---------------------------------------------------------------
    takes = result.takes
    x = np.arange(len(takes))
    rms = np.array([t.rms_db for t in takes])
    err_ax.bar(x, rms, width=0.7, color=SERIES[0], linewidth=0, zorder=3)
    err_ax.axhline(GOOD_FIT_DB, color=MUTED, linewidth=1, zorder=2)
    err_ax.annotate(
        f"{GOOD_FIT_DB:g} dB target", (1, GOOD_FIT_DB), xycoords=("axes fraction", "data"),
        xytext=(0, 3), textcoords="offset points", ha="right", color=INK_SECONDARY, fontsize=8,
    )  # fmt: skip
    _label_clips(err_ax, takes)
    err_ax.set_xlim(-0.8, len(takes) - 0.2)
    err_ax.set_ylabel("RMS error (dB)")
    within = np.mean(rms <= GOOD_FIT_DB) * 100
    err_ax.set_title(
        f"Model vs measurement per take: {within:.0f}% within {GOOD_FIT_DB:g} dB, "
        f"overall {result.rms_db:.2f} dB"
    )

    # --- Pluck position check ---------------------------------------------------------
    shift = np.array([t.fitted_mm - t.manifest_mm for t in takes])
    pos_ax.axhspan(-MAX_BETA_SHIFT_MM, MAX_BETA_SHIFT_MM, color=GRID, alpha=0.5, zorder=0)
    pos_ax.axhline(0, color=MUTED, linewidth=1)
    at_limit = np.abs(shift) >= MAX_BETA_SHIFT_MM - 0.05
    pos_ax.plot(
        x[~at_limit], shift[~at_limit], "o", markersize=7,
        color=SERIES[0], mec=SURFACE, mew=1.5, label="fitted − manifest",
    )  # fmt: skip
    if at_limit.any():
        pos_ax.plot(
            x[at_limit], shift[at_limit], "o", markersize=7, color=SERIES[1], mec=SURFACE,
            mew=1.5, label=f"at the ±{MAX_BETA_SHIFT_MM:g} mm limit: check the tape",
        )  # fmt: skip
    _label_clips(pos_ax, takes)
    pos_ax.set_xlim(-0.8, len(takes) - 0.2)
    pos_ax.set_ylim(-MAX_BETA_SHIFT_MM - 1, MAX_BETA_SHIFT_MM + 1)
    pos_ax.set_ylabel("Pluck position shift (mm)")
    pos_ax.set_title("Fitted pluck position against the manifest (band: allowed range)")
    pos_ax.legend(loc="upper right")

    widths = ", ".join(f"{f} {w:.1f} mm" for f, w in profile.finger_width_mm.items())
    fig.suptitle(
        f"Calibration '{profile.name}': {len(takes)} takes · finger width {widths}",
        x=0.01, ha="left", fontweight="bold", fontsize=13,
    )  # fmt: skip
    return fig


def _label_clips(ax, takes) -> None:
    """One tick per clip, at its first take."""
    starts, labels = [], []
    for i, take in enumerate(takes):
        if i == 0 or take.clip != takes[i - 1].clip:
            starts.append(i)
            labels.append(take.clip)
    ax.set_xticks(starts, labels, rotation=60, ha="right", fontsize=7)
