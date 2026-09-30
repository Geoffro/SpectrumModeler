"""Tuner window: compare the calibrated model with each take and adjust it by hand.

Sliders change the profile's manual values for the current string (damping,
stiffness, finger width, per-harmonic offsets). The pluck-position slider only
shifts the view, to see how the model responds; it is not saved.
"""

import numpy as np
from matplotlib.widgets import Button, Slider

from ..tuning import TunerState
from .style import INK_SECONDARY, MUTED, SERIES, SURFACE, use_style

FLOOR_DB = -60.0


class TunerApp:
    def __init__(self, state: TunerState) -> None:
        import matplotlib.pyplot as plt

        use_style()  # the window is drawn later, outside any temporary style
        self.state = state
        self._syncing = False
        self.fig = plt.figure(figsize=(13, 8.5))
        self.ax = self.fig.add_axes((0.06, 0.45, 0.9, 0.47))

        def slider(row: int, label: str, lo: float, hi: float, **kw) -> Slider:
            ax = self.fig.add_axes((0.2, 0.33 - row * 0.045, 0.4, 0.03))
            widget = Slider(ax, label, lo, hi, color=SERIES[0], **kw)
            widget.label.set_x(-0.03)  # clear of the track
            return widget

        self.position = slider(0, "Position shift (mm)", -10, 10, valinit=0, valstep=0.5)
        self.width = slider(1, "Finger width (mm)", 0, 30, valinit=0, valstep=0.5)
        self.sigma0 = slider(2, "σ0 (1/s)", 0, 5, valinit=1)
        self.sigma2 = slider(3, "σ2 (×10⁻⁶)", 0, 10, valinit=2)
        self.log_b = slider(4, "log10 B", -7, -3, valinit=-5)
        self.harmonic = slider(5, "Harmonic", 1, 2, valinit=1, valstep=1)
        self.offset = slider(6, "Offset (dB)", -20, 20, valinit=0, valstep=0.5)

        self.position.on_changed(self._on_position)
        self.width.on_changed(self._guard(lambda v: state.set_finger_width(v)))
        self.sigma0.on_changed(self._guard(lambda v: state.set_damping(sigma0=v)))
        self.sigma2.on_changed(self._guard(lambda v: state.set_damping(sigma2=v * 1e-6)))
        self.log_b.on_changed(self._guard(lambda v: state.set_inharmonicity(10**v)))
        self.harmonic.on_changed(self._on_harmonic)
        self.offset.on_changed(self._guard(lambda v: state.set_offset(int(self.harmonic.val), v)))

        self.buttons = {}
        actions = [
            ("< Previous", lambda: self._select(-1)),
            ("Next >", lambda: self._select(1)),
            ("Match harmonic", self._match),
            ("Clear string", self._clear),
            ("Play recording", lambda: self._play(0)),
            ("Play model", lambda: self._play(1)),
            ("Save", self._save),
        ]
        for i, (label, action) in enumerate(actions):
            ax = self.fig.add_axes((0.68 + (i % 2) * 0.15, 0.30 - (i // 2) * 0.06, 0.13, 0.045))
            button = Button(ax, label, color=SURFACE, hovercolor="#e1e0d9")
            button.on_clicked(lambda _event, action=action: action())
            self.buttons[label] = button

        self.status = self.fig.text(0.06, 0.015, "", color=INK_SECONDARY, fontsize=9)
        self._sync_sliders()
        self.redraw()

    # --- Callbacks ----------------------------------------------------------------------

    def _guard(self, apply):
        def callback(value):
            if not self._syncing:
                apply(value)
                self.redraw()

        return callback

    def _on_position(self, value) -> None:
        self.state.position_shift_mm = value
        self.redraw()

    def _on_harmonic(self, value) -> None:
        self._syncing = True
        self.offset.set_val(self.state.offset(int(value)))
        self._syncing = False
        self.redraw()

    def _select(self, delta: int) -> None:
        self.state.step(delta)
        self._sync_sliders()
        self.redraw()

    def _match(self) -> None:
        try:
            db = self.state.match_harmonic(int(self.harmonic.val))
            self._sync_sliders()
            self.redraw(f"harmonic {int(self.harmonic.val)} offset set to {db:+.1f} dB")
        except ValueError as e:
            self.redraw(str(e))

    def _clear(self) -> None:
        self.state.clear_string()
        self._sync_sliders()
        self.redraw(f"manual adjustments for {self.state.note} cleared")

    def _play(self, which: int) -> None:
        try:
            from ..audio import play

            audio = self.state.ab_audio()
            play(audio[which], audio[2], blocking=False)
        except Exception as e:  # no session, no audio device, ...
            self.redraw(f"can't play: {e}")

    def _save(self) -> None:
        path = self.state.save()
        self.redraw(f"saved {path}")

    # --- Drawing ------------------------------------------------------------------------------

    def _sync_sliders(self) -> None:
        """Show the current string's effective values without recording them as changes."""
        s = self.state
        self._syncing = True
        sigma0, sigma2 = s.damping()
        self.position.set_val(s.position_shift_mm)
        self.width.set_val(s.finger_width())
        self.sigma0.set_val(sigma0)
        self.sigma2.set_val(sigma2 * 1e6)
        b = s.inharmonicity()
        self.log_b.set_val(np.log10(b) if b else -7)
        n_max = int(s.current.harmonics[-1])
        self.harmonic.valmax = n_max
        self.harmonic.ax.set_xlim(1, n_max)
        self.offset.set_val(s.offset(int(min(self.harmonic.val, n_max))))
        self._syncing = False

    def redraw(self, message: str = "") -> None:
        s, ax = self.state, self.ax
        harmonics, measured, model, reliable = s.levels()
        ref = np.nanmax(measured)
        ax.clear()
        ax.bar(
            harmonics[reliable], measured[reliable] - ref - FLOOR_DB, bottom=FLOOR_DB,
            width=0.6, color=SERIES[0], linewidth=0, label="measured", zorder=3,
        )  # fmt: skip
        ax.plot(
            harmonics, np.maximum(model - ref, FLOOR_DB), "o", markersize=6,
            color=SERIES[1], mec=SURFACE, mew=1, label="calibrated model", zorder=4,
        )  # fmt: skip
        selected = int(self.harmonic.val)
        ax.axvline(selected, color=MUTED, linewidth=1, zorder=2)
        ax.set_xlim(0.3, harmonics[-1] + 0.7)
        ax.set_ylim(FLOOR_DB, 5)
        ax.set_xlabel("Harmonic number")
        ax.set_ylabel("Level at the pluck (dB re strongest)")
        ax.legend(loc="upper right")
        m = s.current
        shift = f" (viewing {s.position_shift_mm:+g} mm)" if s.position_shift_mm else ""
        ax.set_title(
            f"{m.clip} take {m.take} · {m.pluck_mm:g} mm{shift} · "
            f"RMS error {s.rms_error_db():.2f} dB · take {s.index + 1} of {len(s.takes)}"
            + (" · unsaved changes" if s.dirty else ""),
            loc="left",
        )
        manual = s.profile.manual.get(s.note)
        adjusted = manual.describe() if manual and not manual.is_empty() else "none"
        self.status.set_text(
            (message + "   ·   " if message else "") + f"{s.note} manual adjustments: {adjusted}"
        )
        self.fig.canvas.draw_idle()
