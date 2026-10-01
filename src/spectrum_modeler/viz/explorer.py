"""Explorer window: drag the pluck point and watch (and hear) the overtones change."""

import numpy as np
from matplotlib.figure import Figure
from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider

from ..explorer import DISPLAY_HARMONICS, EDGE_MM, ExplorerState
from ..modal import envelope_nulls, initial_shape
from ..presets import FINGER_WIDTHS_MM, OPEN_STRING_NOTES
from ..recording import SAMPLE_RATE
from .shape import frame_string_axes
from .spectrum import describe, node_markers
from .style import C, button_props, check_props, radio_props, slider_props, use_style

FLOOR_DB = -60.0

LAYERS = {"damping": "damping", "finger width": "finger_width",
          "stiffness": "inharmonicity", "body": "body"}  # fmt: skip


class ExplorerApp:
    def __init__(self, state: ExplorerState, fig: Figure | None = None) -> None:
        """Draw in ``fig`` (already attached to a canvas), or in a new pyplot window."""
        import matplotlib.pyplot as plt

        use_style()  # the window is drawn later, outside any temporary style
        self.state = state
        s = state.settings
        self.fig = fig if fig is not None else plt.figure(figsize=(14, 8.5))
        self.shape_ax = self.fig.add_axes((0.06, 0.72, 0.6, 0.2))
        self.spectrum_ax = self.fig.add_axes((0.06, 0.25, 0.6, 0.38))

        def panel(rect, title):
            ax = self.fig.add_axes(rect)
            ax.set_title(title, fontsize=9, loc="left", color=C.INK_SECONDARY)
            ax.set_facecolor(C.SURFACE)
            return ax

        self.strings = RadioButtons(
            panel((0.72, 0.6, 0.1, 0.3), "String"),
            OPEN_STRING_NOTES,
            active=OPEN_STRING_NOTES.index(s.note),
            activecolor=C.SERIES[0],
            radio_props=radio_props(),
        )
        fingers = list(FINGER_WIDTHS_MM)
        self.fingers = RadioButtons(
            panel((0.85, 0.72, 0.12, 0.18), "Finger"),
            fingers,
            active=fingers.index(s.finger),
            activecolor=C.SERIES[0],
            radio_props=radio_props(),
        )
        self.layers = CheckButtons(
            panel((0.85, 0.44, 0.12, 0.2), "Realism layers"),
            list(LAYERS),
            [getattr(s, attr) for attr in LAYERS.values()],
            **check_props(),
        )
        label = "use profile" if state.profile else "no profile loaded"
        self.calibrated = CheckButtons(
            panel((0.72, 0.44, 0.11, 0.1), "Calibration"), [label], [s.calibrated], **check_props()
        )
        self.position = Slider(
            self.fig.add_axes((0.2, 0.14, 0.46, 0.03)),
            "Pluck distance from saddle (mm)",
            EDGE_MM,
            state.max_position_mm,
            valinit=s.position_mm,
            valstep=1,
            **slider_props(),
        )
        self.position.label.set_x(-0.03)

        self.strings.on_clicked(lambda note: self._change(note=note))
        self.fingers.on_clicked(lambda finger: self._change(finger=finger))
        self.layers.on_clicked(lambda name: self._change(**{LAYERS[name]: self._checked(name)}))
        self.calibrated.on_clicked(self._on_calibrated)
        self.position.on_changed(lambda mm: self._change(position_mm=mm))

        self.buttons = {}
        actions = [
            ("Play model", lambda: self._play(state.playable)),
            ("Play recording", lambda: self._play(state.recording_audio)),
            ("Pin as B", self._pin),
            ("Play A then B", lambda: self._play(state.ab_audio)),
            ("Clear B", self._unpin),
        ]
        for i, (text, action) in enumerate(actions):
            ax = self.fig.add_axes((0.06 + i * 0.125, 0.04, 0.115, 0.05))
            button = Button(ax, text, **button_props(ax))
            button.on_clicked(lambda _event, action=action: action())
            self.buttons[text] = button

        self.readout = self.fig.text(
            0.72, 0.38, "", va="top", fontsize=9, color=C.INK, linespacing=1.6
        )
        self.status = self.fig.text(0.72, 0.06, "", fontsize=9, color=C.INK_SECONDARY)
        self._build()
        self.redraw()

    # --- Callbacks ---------------------------------------------------------------------

    def _checked(self, name: str) -> bool:
        return self.layers.get_status()[list(LAYERS).index(name)]

    def _change(self, **changes) -> None:
        self.state.update(**changes)
        self.redraw()

    def _on_calibrated(self, _label) -> None:
        on = self.calibrated.get_status()[0]
        if on and self.state.profile is None:
            self.calibrated.set_active(0)  # untick
            self.redraw("open a profile to compare against your guitar")
            return
        self.state.update(calibrated=on)
        self._sync_layers()
        self.redraw()

    def _sync_layers(self) -> None:
        """Tick boxes to match the state (calibrating switches stiffness and body on)."""
        s = self.state.settings
        for i, (name, attr) in enumerate(LAYERS.items()):
            if self._checked(name) != getattr(s, attr):
                self.layers.eventson = False
                self.layers.set_active(i)
                self.layers.eventson = True

    def _pin(self) -> None:
        self.state.pin()
        self.redraw(f"pinned B: {self.state.pinned.describe()}")

    def _unpin(self) -> None:
        self.state.unpin()
        self.redraw()

    def _play(self, make_audio) -> None:
        try:
            from ..audio import play

            play(make_audio(), SAMPLE_RATE, blocking=False)
        except Exception as e:  # nothing pinned, no recording, no audio device, ...
            self.redraw(f"can't play: {e}")

    # --- Drawing -------------------------------------------------------------------------------
    # Artists are built once and updated in place: clearing and replotting the
    # axes on every slider step is several times slower.

    def _build(self) -> None:
        state = self.state
        spectrum = state.spectrum()

        ax = self.shape_ax
        (self.shape_line,) = ax.plot([], [], color=C.SERIES[0], zorder=3)
        (self.pluck_dot,) = ax.plot(
            [], [], "o", markersize=8, color=C.SERIES[0], mec=C.SURFACE, mew=2, zorder=4
        )
        self.pluck_label = ax.annotate(
            "", (0, 0), xytext=(8, 4), textcoords="offset points", color=C.INK_SECONDARY, fontsize=9
        )
        frame_string_axes(ax, spectrum.string, spectrum.pluck)

        ax = self.spectrum_ax
        n = np.arange(1, DISPLAY_HARMONICS + 1)
        self.bars = ax.bar(
            n, np.zeros(n.size), bottom=FLOOR_DB, width=0.6,
            color=C.SERIES[0], linewidth=0, zorder=3, label="A",
        )  # fmt: skip
        self.pinned_dots = ax.plot(
            [], [], "o", markersize=6, color=C.SERIES[1], mec=C.SURFACE, mew=1, zorder=5, label="B"
        )[0]
        self.nodes = node_markers(ax, np.array([]))
        ax.set_xlim(0.3, DISPLAY_HARMONICS + 0.7)
        ax.set_ylim(FLOOR_DB, 5)
        ax.set_xticks([1, *range(5, DISPLAY_HARMONICS + 1, 5)])
        ax.set_xlabel("Harmonic number")
        ax.set_ylabel("Bridge force (dB re strongest)")

    def redraw(self, message: str = "") -> None:
        state = self.state
        spectrum = state.spectrum()
        pinned = state.pinned_spectrum()
        string, pluck = spectrum.string, spectrum.pluck

        x = np.union1d(
            np.linspace(0, string.length, 401), [pluck.beta(string.length) * string.length]
        )
        self.shape_line.set_data(x * 1000, initial_shape(x, string.length, pluck) * 1000)
        self.pluck_dot.set_data([pluck.distance_mm], [pluck.amplitude * 1000])
        self.pluck_label.xy = (pluck.distance_mm, pluck.amplitude * 1000)
        self.pluck_label.set_text(f"pluck, {pluck.distance_mm:g} mm")
        self.shape_ax.set_xlim(0, string.length * 1000)
        self.shape_ax.set_title(describe(spectrum))

        ref = np.abs(spectrum.amps).max()
        if pinned is not None:
            ref = max(ref, np.abs(pinned.amps).max())
        for bar, level in zip(
            self.bars, spectrum.levels_db(ref=ref, floor_db=FLOOR_DB), strict=True
        ):
            bar.set_height(level - FLOOR_DB)
        self.nodes.set_xdata(envelope_nulls(spectrum.beta, DISPLAY_HARMONICS + 0.5))
        self.nodes.set_ydata(np.full(self.nodes.get_xdata().size, 0.985))
        if pinned is not None:
            self.pinned_dots.set_data(
                pinned.harmonics, pinned.levels_db(ref=ref, floor_db=FLOOR_DB)
            )
        else:
            self.pinned_dots.set_data([], [])
        self._legend(pinned is not None)

        self.readout.set_text("\n".join(state.readout()))
        self.status.set_text(message)
        self.fig.canvas.draw_idle()

    def _legend(self, with_pinned: bool) -> None:
        s = self.state
        handles = [self.bars, self.nodes]
        labels = [f"A: {s.settings.describe()}", "node at pluck point"]
        if with_pinned:
            handles.insert(1, self.pinned_dots)
            labels.insert(1, f"B: {s.pinned.describe()}")
        self.spectrum_ax.legend(handles, labels, loc="upper right", fontsize=8)
