"""The SpectrumModeler app: explorer, pluck comparison, and tuner in tabs.

One dark-themed Tk window. The Explorer and Tuner tabs are the matplotlib
windows from ``viz/explorer.py`` and ``viz/tuner.py``, embedded; the Compare tab
has its controls in a sidebar. Files exported from the app match what
``compare_positions.py`` writes (light theme).

A recording session and a guitar profile are opened from the bar at the top,
and remembered for next time.
"""

import json
import tkinter as tk
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from ..analysis import spectral_centroid, spectrogram
from ..audio import match_loudness, sequence, write_wav
from ..explorer import EDGE_MM, ExplorerState, Settings
from ..measure import load_session_measurements
from ..presets import FINGER_WIDTHS_MM, OPEN_STRING_NOTES, PLUCK_PRESETS, pluck_position_mm
from ..profile import PROFILE_DIR, load_profile
from ..recording import SAMPLE_RATE, Session
from ..string_sets import DEFAULT_STRING_SET
from ..tuning import TunerState
from . import style
from .explorer import LAYERS, ExplorerApp
from .shape import plot_mode_decomposition
from .spectrogram import plot_spectrogram_pair
from .spectrum import describe, plot_comparison
from .style import C
from .tuner import TunerApp

TABS = {"explore": "Explorer", "compare": "Compare", "tune": "Tuner"}
COMPARE_SECONDS = 4.0  # audio for spectrograms, playback, and export
EXPORT_DIR = Path("output")
CONFIG_PATH = Path.home() / ".config" / "spectrum-modeler" / "app.json"


# --- Shared pieces ----------------------------------------------------------------------------


def _style_tk(root: tk.Tk) -> None:
    """Dark ttk widgets to match the charts."""
    root.configure(bg=C.SURFACE)
    try:  # dark title bar on macOS (Tk 8.6.10+)
        root.tk.call("::tk::unsupported::MacWindowStyle", "appearance", root, "darkaqua")
    except tk.TclError:
        pass
    s = ttk.Style(root)
    s.theme_use("clam")  # the native macOS theme ignores most colors
    s.configure(
        ".", background=C.SURFACE, foreground=C.INK, fieldbackground=C.GRID,
        bordercolor=C.BASELINE, lightcolor=C.GRID, darkcolor=C.GRID, troughcolor=C.GRID,
        selectbackground=C.SERIES[0], selectforeground=C.INK, insertcolor=C.INK,
        arrowcolor=C.INK, focuscolor=C.SERIES[0],
    )  # fmt: skip
    s.map(".", foreground=[("disabled", C.MUTED)])
    s.configure("TNotebook", borderwidth=0, tabmargins=(8, 6, 8, 0))
    s.configure("TNotebook.Tab", padding=(18, 6), borderwidth=0, foreground=C.INK_SECONDARY)
    s.map(
        "TNotebook.Tab",
        background=[("selected", C.GRID), ("active", C.GRID)],
        foreground=[("selected", C.INK)],
    )
    s.configure("TButton", background=C.GRID, padding=(10, 4))
    s.map("TButton", background=[("pressed", C.BASELINE), ("active", C.BASELINE)])
    for widget in ("TCheckbutton", "TRadiobutton"):
        s.configure(widget, indicatorbackground=C.GRID, indicatorforeground=C.INK)
        s.map(
            widget,
            background=[("active", C.SURFACE)],
            indicatorbackground=[("selected", C.SERIES[0]), ("disabled", C.SURFACE)],
        )
    s.configure("TCombobox", background=C.GRID)
    s.map(
        "TCombobox",
        fieldbackground=[("readonly", C.GRID)],
        selectbackground=[("readonly", C.GRID)],
        selectforeground=[("readonly", C.INK)],
    )
    s.configure("TSpinbox", background=C.GRID)
    s.configure("Heading.TLabel", foreground=C.INK_SECONDARY)
    s.configure("Status.TLabel", foreground=C.INK_SECONDARY)
    listbox = {"background": C.GRID, "foreground": C.INK,
               "selectBackground": C.SERIES[0], "selectForeground": C.INK}  # fmt: skip
    for option, color in listbox.items():
        root.option_add(f"*TCombobox*Listbox.{option}", color)


def _embed(parent: tk.Widget) -> Figure:
    """A figure filling ``parent`` (to the right of any sidebar)."""
    fig = Figure(figsize=(11, 8))
    canvas = FigureCanvasTkAgg(fig, master=parent)
    widget = canvas.get_tk_widget()
    widget.configure(bg=C.SURFACE, highlightthickness=0)
    widget.pack(side="right", fill="both", expand=True)
    return fig


def _play(audio: np.ndarray) -> None:
    from ..audio import play

    play(audio, SAMPLE_RATE, blocking=False)


def _slug(text: str, mm: float) -> str:
    """A position for file names, as the scripts write it: the preset, or ``40mm``."""
    return text if text in PLUCK_PRESETS else f"{mm:g}mm"


class Sidebar(ttk.Frame):
    """A column of labelled controls; every change calls ``on_change``."""

    def __init__(self, parent: tk.Widget, on_change: Callable[[], None]) -> None:
        super().__init__(parent, padding=(16, 12), width=230)
        self.pack(side="left", fill="y")
        self.on_change = on_change

    def _watch(self, var: tk.Variable) -> tk.Variable:
        var.trace_add("write", lambda *_: self.on_change())
        return var

    def heading(self, text: str) -> None:
        ttk.Label(self, text=text, style="Heading.TLabel").pack(anchor="w", pady=(10, 2))

    def combo(self, label: str, values, value: str, *, editable: bool = False) -> tk.StringVar:
        self.heading(label)
        var = tk.StringVar(value=value)
        ttk.Combobox(
            self, textvariable=var, values=list(values), width=16,
            state="normal" if editable else "readonly",
        ).pack(anchor="w", fill="x")  # fmt: skip
        return self._watch(var)

    def check(self, label: str, value: bool, *, enabled: bool = True) -> tk.BooleanVar:
        var = tk.BooleanVar(value=value)
        ttk.Checkbutton(
            self, text=label, variable=var, state="normal" if enabled else "disabled"
        ).pack(anchor="w")
        return self._watch(var)

    def spin(self, label: str, lo: float, hi: float, step: float, value: float) -> tk.DoubleVar:
        self.heading(label)
        var = tk.DoubleVar(value=value)
        box = ttk.Spinbox(
            self, textvariable=var, from_=lo, to=hi, increment=step, width=8,
            command=self.on_change,
        )  # fmt: skip
        box.pack(anchor="w")
        for event in ("<Return>", "<FocusOut>"):  # typed values: not on every keystroke
            box.bind(event, lambda _event: self.on_change())
        return var

    def radios(self, label: str, options, value: str) -> tk.StringVar:
        self.heading(label)
        var = tk.StringVar(value=value)
        for option in options:
            ttk.Radiobutton(self, text=option, value=option, variable=var).pack(anchor="w")
        return self._watch(var)

    def buttons(self, actions: list[tuple[str, Callable[[], None]]]) -> dict[str, ttk.Button]:
        frame = ttk.Frame(self)
        frame.pack(anchor="w", fill="x", pady=(16, 0))
        made = {}
        for i, (text, action) in enumerate(actions):
            button = ttk.Button(frame, text=text, command=action)
            button.grid(row=i // 2, column=i % 2, sticky="ew", padx=(0, 4), pady=2)
            made[text] = button
        frame.columnconfigure((0, 1), weight=1)
        return made

    def status(self) -> tk.StringVar:
        var = tk.StringVar()
        ttk.Label(self, textvariable=var, style="Status.TLabel", wraplength=210).pack(
            anchor="w", side="bottom", pady=(8, 0)
        )
        return var


class _Tab:
    """Redraws are coalesced: a burst of changes redraws once, shortly after."""

    def __init__(self, parent: tk.Widget) -> None:
        self.parent = parent
        self._job: str | None = None

    def schedule(self, delay_ms: int = 150) -> None:
        if self._job is not None:
            self.parent.after_cancel(self._job)
        self._job = self.parent.after(delay_ms, self._run)

    def cancel(self) -> None:
        """Drop a pending redraw (before the tab is destroyed)."""
        if self._job is not None:
            self.parent.after_cancel(self._job)
            self._job = None

    def _run(self) -> None:
        self._job = None
        top = self.parent.winfo_toplevel()
        top.configure(cursor="watch")
        top.update_idletasks()
        try:
            self.redraw()
        except ValueError as e:  # a half-typed position, and so on
            self.status.set(str(e))
        finally:
            top.configure(cursor="")

    def position_mm(self, label: str, text: str, state: ExplorerState) -> float:
        """A preset name or mm, kept on the string."""
        text = text.strip()
        try:
            mm = pluck_position_mm(text if text in PLUCK_PRESETS else float(text), state.length)
        except ValueError:
            raise ValueError(f"{label}: give mm, or one of {', '.join(PLUCK_PRESETS)}") from None
        return float(np.clip(mm, EDGE_MM, state.max_position_mm))

    def redraw(self) -> None:
        raise NotImplementedError


# --- Compare ------------------------------------------------------------------------------------


class CompareTab(_Tab):
    """Two pluck positions side by side: spectra, spectrograms, or mode decompositions."""

    VIEWS = ("spectra", "spectrograms", "modes of A", "modes of B")

    def __init__(self, parent: tk.Widget, state: ExplorerState) -> None:
        super().__init__(parent)
        self.state = state
        side = Sidebar(parent, self.schedule)
        self.fig = _embed(parent)
        s = state.settings
        self.note = side.combo("String", OPEN_STRING_NOTES, s.note)
        self.a = side.combo("Pluck A (preset or mm)", PLUCK_PRESETS, "ponticello", editable=True)
        self.b = side.combo("Pluck B (preset or mm)", PLUCK_PRESETS, "tasto", editable=True)
        self.finger = side.combo("Finger", FINGER_WIDTHS_MM, s.finger)
        side.heading("Realism layers")
        self.layers = {attr: side.check(name, getattr(s, attr)) for name, attr in LAYERS.items()}
        side.heading("Calibration")
        self.calibrated = side.check(
            "use profile" if state.profile else "no profile loaded",
            s.calibrated,
            enabled=state.profile is not None,
        )
        self.view = side.radios("View", self.VIEWS, self.VIEWS[0])
        side.buttons([
            ("Play A", lambda: self._play(0)),
            ("Play B", lambda: self._play(1)),
            ("Play A then B", lambda: self._play(2)),
            ("Export…", self.export),
        ])  # fmt: skip
        self.status = side.status()
        self._audio_key = None
        self._audio: tuple[np.ndarray, np.ndarray] | None = None
        self.schedule(0)

    def settings(self) -> tuple[Settings, Settings]:
        common = dict(
            note=self.note.get(),
            finger=self.finger.get(),
            calibrated=self.calibrated.get(),
            **{attr: var.get() for attr, var in self.layers.items()},
        )
        return (
            Settings(position_mm=self.position_mm("Pluck A", self.a.get(), self.state), **common),
            Settings(position_mm=self.position_mm("Pluck B", self.b.get(), self.state), **common),
        )

    def audio(self) -> tuple[np.ndarray, np.ndarray]:
        """Both plucks, matched in loudness (kept until the settings change)."""
        key = self.settings()
        if key != self._audio_key:
            a, b = key
            self._audio = tuple(
                match_loudness(
                    self.state.audio(a, COMPARE_SECONDS), self.state.audio(b, COMPARE_SECONDS)
                )
            )
            self._audio_key = key
        return self._audio

    def draw(self, fig: Figure, view: str) -> Figure:
        a, b = self.settings()
        spec_a, spec_b = self.state.spectrum(a), self.state.spectrum(b)
        if view == "spectra":
            return plot_comparison(spec_a, spec_b, fig=fig)
        if view == "spectrograms":
            audio_a, audio_b = self.audio()
            return plot_spectrogram_pair(
                spectrogram(audio_a, SAMPLE_RATE),
                spectrogram(audio_b, SAMPLE_RATE),
                labels=(describe(spec_a), describe(spec_b)),
                suptitle=f"{a.note}: spectrogram, matched loudness",
                fig=fig,
            )
        spectrum = spec_a if view == "modes of A" else spec_b
        return plot_mode_decomposition(spectrum.string, spectrum.pluck, fig=fig)

    def redraw(self) -> None:
        self.draw(self.fig, self.view.get())
        self.fig.canvas.draw_idle()
        a, b = (self.state.spectrum(s) for s in self.settings())
        self.status.set(
            "brightness (spectral centroid): "
            f"A {spectral_centroid(a.freqs, a.amps):.0f} Hz, "
            f"B {spectral_centroid(b.freqs, b.amps):.0f} Hz"
        )

    def _play(self, which: int) -> None:
        try:
            a, b = self.audio()
            _play((a, b, sequence([a, b], SAMPLE_RATE))[which])
        except Exception as e:  # a bad position, no audio device, ...
            self.status.set(f"can't play: {e}")

    def export(self, out: Path | None = None) -> list[Path]:
        """Write what compare_positions.py writes for this string and these settings."""
        if out is None:
            chosen = filedialog.askdirectory(
                title="Export comparison to…", initialdir=EXPORT_DIR.resolve(), mustexist=False
            )
            if not chosen:
                return []
            out = Path(chosen)
        written = []
        try:
            a, b = self.settings()
            note, slug_a, slug_b = (
                a.note,
                _slug(self.a.get().strip(), a.position_mm),
                _slug(self.b.get().strip(), b.position_mm),
            )
            out.mkdir(parents=True, exist_ok=True)
            stem = f"{note}_{slug_a}_vs_{slug_b}"
            figures = {
                f"{stem}.png": "spectra",
                f"{stem}_spectrogram.png": "spectrograms",
                f"{note}_{slug_a}_modes.png": "modes of A",
                f"{note}_{slug_b}_modes.png": "modes of B",
            }
            with style.using_theme("light"):
                for name, view in figures.items():
                    self.draw(None, view).savefig(out / name)
                    written.append(out / name)
            audio_a, audio_b = self.audio()
            for name, audio in (
                (f"{note}_{slug_a}.wav", audio_a),
                (f"{note}_{slug_b}.wav", audio_b),
                (f"{note}_{slug_a}_then_{slug_b}.wav", sequence([audio_a, audio_b], SAMPLE_RATE)),
            ):
                write_wav(out / name, audio, SAMPLE_RATE)
                written.append(out / name)
        except Exception as e:  # a bad position, a full disk, ...
            self.status.set(f"export failed: {e}")
            return written
        self.status.set(f"wrote {len(written)} files to {out}")
        return written


# --- What is open ------------------------------------------------------------------------------


@dataclass(frozen=True)
class Workspace:
    """The open recording session and guitar profile, either optional."""

    session: Path | None = None
    profile: Path | None = None

    @classmethod
    def recall(cls, config: Path | None) -> "Workspace":
        """What was open last time (nothing, if ``config`` is missing or unreadable)."""
        if config is None:
            return cls()
        try:
            data = json.loads(config.read_text())
            return cls(*(Path(data[k]) if data.get(k) else None for k in ("session", "profile")))
        except (OSError, ValueError, AttributeError, TypeError):  # missing, not JSON, not ours
            return cls()

    def remember(self, config: Path | None) -> None:
        if config is None:
            return
        config.parent.mkdir(parents=True, exist_ok=True)
        data = {k: str(v.resolve()) if v else None for k, v in vars(self).items()}
        config.write_text(json.dumps(data, indent=2))


@dataclass
class Loaded:
    workspace: Workspace  # without whatever failed to open
    explorer: ExplorerState
    tuner: TunerState | None
    tuner_hint: str
    n_takes: int = 0
    problems: list[str] = field(default_factory=list)


TUNER_NEEDS = (
    "The tuner compares a calibrated profile with measured takes.\n"
    "Open an analyzed session and a profile with the buttons above."
)


def load_workspace(workspace: Workspace) -> Loaded:
    """Open the session and profile; what can't be opened is left out and reported."""
    problems = []
    profile = None
    if workspace.profile:
        try:
            profile = load_profile(workspace.profile)
        except Exception as e:  # a chosen file: missing, not JSON, not a profile, ...
            problems.append(f"Couldn't open profile {workspace.profile.name}: {e}")
            workspace = replace(workspace, profile=None)
    session = measurements = None
    if workspace.session:
        try:
            session = Session(workspace.session)
            measurements = load_session_measurements(session)
        except Exception as e:  # not a session, or not analyzed yet
            problems.append(f"Couldn't open session {workspace.session.name}: {e}")
            workspace = replace(workspace, session=None)
            session = None

    tuner, hint = None, TUNER_NEEDS
    if profile and session:
        try:
            tuner = TunerState(profile, measurements, workspace.profile, session)
        except ValueError as e:
            hint = f"Tuner unavailable: {e}"
    explorer = ExplorerState(
        string_set=profile.string_set if profile else DEFAULT_STRING_SET,
        profile=profile,
        session=session,
        measurements=measurements,
    )
    return Loaded(workspace, explorer, tuner, hint, len(measurements or []), problems)


def _short(path: Path) -> str:
    """A path as short as it can be shown: relative to here, or from home."""
    path = path.resolve()
    for base, prefix in ((Path.cwd(), ""), (Path.home(), "~/")):
        if path.is_relative_to(base):
            return prefix + str(path.relative_to(base))
    return str(path)


# --- Window -------------------------------------------------------------------------------------


class App:
    """The main window. Opens what was open last time (from ``config``; None to forget)."""

    def __init__(
        self, root: tk.Tk, *, config: Path | None = CONFIG_PATH, tab: str = "explore"
    ) -> None:
        style.set_theme("dark")
        style.use_style()
        style.use_pyplot(False)
        self.root = root
        self.config = config
        root.title("SpectrumModeler")
        root.geometry("1440x900")
        _style_tk(root)

        bar = ttk.Frame(root, padding=(12, 8, 12, 0))
        bar.pack(side="top", fill="x")
        self.session_text, self.profile_text = tk.StringVar(), tk.StringVar()
        for text, var, choose, close in (
            ("Open session…", self.session_text, self.choose_session, self.close_session),
            ("Open profile…", self.profile_text, self.choose_profile, self.close_profile),
        ):
            ttk.Button(bar, text=text, command=choose).pack(side="left")
            ttk.Label(bar, textvariable=var).pack(side="left", padx=(8, 2))
            ttk.Button(bar, text="×", width=2, command=close).pack(side="left", padx=(0, 24))
        self.problems = tk.StringVar()
        ttk.Label(bar, textvariable=self.problems, style="Status.TLabel").pack(side="left")

        self.notebook: ttk.Notebook | None = None
        self.tuner: TunerApp | None = None
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.workspace = Workspace.recall(config)
        self._load(tab)

    # --- Opening sessions and profiles ---------------------------------------------------------

    def _load(self, tab: str | None = None) -> None:
        """Open the workspace and rebuild the tabs for it, on ``tab`` (or the current one)."""
        if tab is None and self.notebook is not None:
            tab = list(TABS)[self.notebook.index("current")]
        top = self.root
        top.configure(cursor="watch")
        top.update_idletasks()
        try:
            loaded = load_workspace(self.workspace)
            self.workspace = loaded.workspace
            self.workspace.remember(self.config)
            self._build(loaded, tab or "explore")
        finally:
            top.configure(cursor="")
        w = self.workspace
        self.session_text.set(
            f"{_short(w.session)} · {loaded.n_takes} takes" if w.session else "no session"
        )
        self.profile_text.set(_short(w.profile) if w.profile else "no profile")
        self.problems.set("   ".join(loaded.problems))

    def _build(self, loaded: Loaded, tab: str) -> None:
        if self.notebook is not None:
            self.compare.cancel()
            self.notebook.destroy()
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True)
        frames = {}
        for key, title in TABS.items():
            frames[key] = ttk.Frame(self.notebook)
            self.notebook.add(frames[key], text=title)

        explorer = loaded.explorer
        self.explorer = ExplorerApp(explorer, fig=_embed(frames["explore"]))
        self.compare = CompareTab(frames["compare"], explorer)
        self.tuner = None
        if loaded.tuner is not None:
            self.tuner = TunerApp(loaded.tuner, fig=_embed(frames["tune"]))
        else:
            fig = _embed(frames["tune"])
            fig.text(0.5, 0.5, loaded.tuner_hint, ha="center", va="center",
                     color=C.INK_SECONDARY, fontsize=11, linespacing=1.8)  # fmt: skip

        self.notebook.select(list(TABS).index(tab))

    def open_session(self, path: Path | None) -> None:
        if self._unsaved_handled():
            self.workspace = replace(self.workspace, session=path)
            self._load()

    def open_profile(self, path: Path | None) -> None:
        if self._unsaved_handled():
            self.workspace = replace(self.workspace, profile=path)
            self._load()

    def close_session(self) -> None:
        if self.workspace.session:
            self.open_session(None)

    def close_profile(self) -> None:
        if self.workspace.profile:
            self.open_profile(None)

    def choose_session(self) -> None:
        start = self.workspace.session or Path("recordings")
        chosen = filedialog.askdirectory(
            title="Open an analyzed recording session",
            initialdir=start.resolve() if start.is_dir() else Path.cwd(),
            mustexist=True,
        )
        if chosen:
            self.open_session(Path(chosen))

    def choose_profile(self) -> None:
        start = self.workspace.profile.parent if self.workspace.profile else PROFILE_DIR
        chosen = filedialog.askopenfilename(
            title="Open a guitar profile",
            initialdir=start.resolve() if start.is_dir() else Path.cwd(),
            filetypes=[("Guitar profile", "*.json")],
        )
        if chosen:
            self.open_profile(Path(chosen))

    def _unsaved_handled(self) -> bool:
        """Offer to save the tuner's changes; False if the user cancels."""
        tuner = self.tuner.state if self.tuner else None
        if tuner is None or not tuner.dirty:
            return True
        save = messagebox.askyesnocancel(
            "Unsaved tuner changes", f"Save your manual adjustments to {tuner.path}?"
        )
        if save:
            tuner.save()
        return save is not None

    # --- Window ----------------------------------------------------------------------------------

    def close(self) -> None:
        if not self._unsaved_handled():
            return
        self.compare.cancel()
        self.root.quit()
        self.root.destroy()
