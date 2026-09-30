"""Time-domain views of the simulated string: animation, filmstrip, and the modal check."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.figure import Figure

from ..fdtd import FdtdResult, harmonic_amplitudes
from ..modal import ModalSpectrum, initial_shape
from .shape import frame_string_axes
from .style import BASELINE, INK_SECONDARY, MUTED, SERIES, SURFACE, styled


def _title(result: FdtdResult) -> str:
    s, p = result.string, result.pluck
    return f"{s.note}, plucked {p.distance_mm:g} mm from the saddle"


@styled
def animate_string(result: FdtdResult, *, slowdown: float = 200.0, fps: int = 30) -> FuncAnimation:
    """The string's motion in slow motion, with the bridge force traced below."""
    if result.snapshots.shape[0] < 2:
        raise ValueError("Simulate with snapshot_every set to get frames to animate")
    s, p = result.string, result.pluck
    fig, (string_ax, force_ax) = plt.subplots(
        2, 1, figsize=(10, 6), height_ratios=[1.4, 1], layout="constrained"
    )

    x_mm = result.x * 1000
    string_ax.plot(
        x_mm, initial_shape(result.x, s.length, p) * 1000,
        color=BASELINE, linewidth=1, label="shape at release",
    )  # fmt: skip
    (line,) = string_ax.plot([], [], color=SERIES[0], label="string")
    frame_string_axes(string_ax, s, p)
    peak = max(np.abs(result.snapshots).max() * 1000, p.amplitude * 1000)
    string_ax.set_ylim(-1.2 * peak, 1.2 * peak)
    string_ax.legend(loc="upper right")
    clock = string_ax.text(
        0.01, 0.06, "", transform=string_ax.transAxes, va="bottom", color=INK_SECONDARY
    )

    end = result.snapshot_times[-1]
    span = result.times <= end
    force_ms, force = result.times[span] * 1000, result.bridge_force[span]
    force_ax.plot(force_ms, force, color=MUTED, linewidth=1)
    (trace,) = force_ax.plot([], [], color=SERIES[1])
    (cursor,) = force_ax.plot([], [], "o", markersize=8, color=SERIES[1], mec=SURFACE, mew=2)
    force_ax.set_xlim(0, end * 1000)
    force_ax.set_xlabel("Time after release (ms)")
    force_ax.set_ylabel("Bridge force (N)")
    force_ax.set_title("What the soundboard feels: force at the bridge", fontsize=10)
    fig.suptitle(
        f"{_title(result)} · {slowdown:g}× slower than real time",
        x=0.01, ha="left", fontweight="bold",
    )  # fmt: skip

    # Lay out once, then freeze: re-solving the layout for every frame is most of
    # the rendering time, and nothing moves that would change it.
    fig.canvas.draw()
    fig.set_layout_engine("none")

    # Frames spaced so the animation runs at `slowdown` times real time.
    frame_times = np.arange(0, end, 1 / (fps * slowdown))
    frame_index = np.searchsorted(result.snapshot_times, frame_times).clip(
        0, len(result.snapshot_times) - 1
    )

    def update(i: int):
        j = frame_index[i]
        t = result.snapshot_times[j]
        line.set_data(x_mm, result.snapshots[j] * 1000)
        upto = force_ms <= t * 1000
        trace.set_data(force_ms[upto], force[upto])
        cursor.set_data([t * 1000], [force[upto][-1]] if upto.any() else [0])
        clock.set_text(f"t = {t * 1000:.2f} ms")
        return line, trace, cursor, clock

    return FuncAnimation(fig, update, frames=len(frame_times), interval=1000 / fps, blit=False)


def save_animation(animation: FuncAnimation, path: str | Path, fps: int = 30) -> Path:
    """Write MP4 (needs ffmpeg) or GIF, by the path's suffix."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gif":
        writer = PillowWriter(fps=fps)
    else:
        writer = FFMpegWriter(fps=fps, codec="h264", extra_args=["-pix_fmt", "yuv420p"])
    animation.save(path, writer=writer, dpi=100)
    plt.close(animation._fig)
    return path


@styled
def plot_filmstrip(result: FdtdResult, n_frames: int = 8) -> Figure:
    """Snapshots through one period: two kinks run along the string and reflect."""
    s = result.string
    period = 1 / s.f0
    times = np.linspace(0, period, n_frames, endpoint=False)
    index = np.searchsorted(result.snapshot_times, times).clip(0, len(result.snapshot_times) - 1)
    fig, axes = plt.subplots(
        n_frames, 1, figsize=(9, 1.1 * n_frames), sharex=True, layout="constrained"
    )
    peak = np.abs(result.snapshots).max() * 1000
    for ax, j in zip(axes, index, strict=True):
        ax.axhline(0, color=BASELINE, linewidth=1)
        ax.plot(result.x * 1000, result.snapshots[j] * 1000, color=SERIES[0])
        ax.set_ylim(-1.2 * peak, 1.2 * peak)
        ax.set_yticks([])
        ax.tick_params(bottom=False)
        ax.grid(False)
        ax.text(
            1.0, 0.5, f"{result.snapshot_times[j] / period:.3f} T",
            transform=ax.transAxes, ha="left", va="center", color=INK_SECONDARY, fontsize=9,
        )  # fmt: skip
        for side in ("left", "bottom"):
            ax.spines[side].set_visible(False)
    axes[-1].spines["bottom"].set_visible(True)
    axes[-1].tick_params(bottom=True)
    axes[-1].set_xlabel("Distance from saddle (mm)")
    axes[-1].set_xlim(0, s.length * 1000)
    fig.suptitle(
        f"{_title(result)}: one period (T = {period * 1000:.2f} ms)",
        x=0.01, ha="left", fontweight="bold",
    )  # fmt: skip
    return fig


@styled
def plot_modal_check(result: FdtdResult, modal: ModalSpectrum, floor_db: float = -60.0) -> Figure:
    """Harmonic levels of the simulated bridge force against the modal model."""
    simulated = harmonic_amplitudes(result, modal.freqs)
    ref = np.abs(modal.amps).max()
    model_db = np.maximum(20 * np.log10(np.abs(modal.amps) / ref + 1e-30), floor_db)
    sim_db = np.maximum(20 * np.log10(simulated / ref + 1e-30), floor_db)

    fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
    n = modal.harmonics
    ax.bar(n, model_db - floor_db, bottom=floor_db, width=0.6, color=SERIES[0], linewidth=0,
           label="modal model", zorder=3)  # fmt: skip
    ax.plot(n, sim_db, "o", markersize=7, color=SERIES[1], mec=SURFACE, mew=1.5,
            label="finite-difference simulation", zorder=4)  # fmt: skip
    usable = np.abs(modal.amps) > ref * 1e-2
    worst = np.abs(sim_db - model_db)[usable].max()
    ax.set_ylim(floor_db, 5)
    ax.set_xlim(0.3, n[-1] + 0.7)
    ax.set_xticks([1, *range(5, n[-1] + 1, 5)])
    ax.set_xlabel("Harmonic number")
    ax.set_ylabel("Bridge force (dB re strongest)")
    ax.set_title(
        f"{_title(result)}: largest difference {worst:.2f} dB "
        f"(harmonics within 40 dB of the strongest)"
    )
    ax.legend(loc="upper right")
    return fig
