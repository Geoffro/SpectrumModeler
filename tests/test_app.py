"""The tabbed app opens sessions and profiles, redraws each tab, and exports what the
scripts write."""

import json
import shutil

import pytest

from spectrum_modeler.calibrate import calibrate
from spectrum_modeler.measure import save_measurements
from spectrum_modeler.recording import Session
from spectrum_modeler.viz import style
from spectrum_modeler.viz.app import Workspace, load_workspace

tk = pytest.importorskip("tkinter")


@pytest.fixture
def root():
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display")
    root.withdraw()
    yield root
    style.set_theme("light")
    style.use_pyplot(True)
    try:
        root.destroy()
    except tk.TclError:  # closed by the test
        pass


def settle(root) -> None:
    """Run pending redraws (they are scheduled with ``after``)."""
    root.after(200)
    root.update()


def test_what_cannot_be_opened_is_reported_and_forgotten(tmp_path):
    config = tmp_path / "app.json"
    Workspace(session=tmp_path / "nowhere", profile=tmp_path / "none.json").remember(config)
    loaded = load_workspace(Workspace.recall(config))
    assert loaded.workspace == Workspace()
    assert len(loaded.problems) == 2 and loaded.tuner is None
    assert Workspace.recall(tmp_path / "missing.json") == Workspace()
    assert Workspace.recall(None) == Workspace()
    for text in ("not json", "[]", '{"session": 3}'):
        config.write_text(text)
        assert Workspace.recall(config) == Workspace()


def test_app_tabs(root, calibration_measurements, tmp_path, monkeypatch):
    from spectrum_modeler.viz.app import TABS, App

    source, measurements = calibration_measurements
    session = Session(tmp_path / "session")  # an analyzed copy, as analyze.py leaves it
    shutil.copytree(source.root, session.root)
    save_measurements(session.measurements / "all.json", measurements)
    profile_file = tmp_path / "synthetic.json"
    calibrate(measurements, name="synthetic").profile.save(profile_file)
    config = tmp_path / "app.json"
    Workspace(session=session.root, profile=profile_file).remember(config)

    app = App(root, config=config)  # reopens what was open last time
    settle(root)
    assert style.current_theme().name == "dark"
    assert app.tuner is not None
    assert f"{len(measurements)} takes" in app.session_text.get()
    assert app.compare.calibrated.get()
    assert app.notebook.index("end") == len(TABS)

    compare = app.compare
    assert len(compare.fig.axes) == 4  # shapes and spectra
    compare.a.set("55")
    compare.view.set("spectrograms")
    settle(root)
    assert "A " in compare.status.get() and "B " in compare.status.get()
    compare.a.set("somewhere")
    settle(root)
    assert "Pluck A" in compare.status.get()
    assert compare.export(tmp_path / "compare") == []
    assert "export failed" in compare.status.get()
    compare.a.set("ponticello")
    written = compare.export(tmp_path / "compare")
    assert {p.name for p in written} == {
        "E2_ponticello_vs_tasto.png",
        "E2_ponticello_vs_tasto_spectrogram.png",
        "E2_ponticello_modes.png",
        "E2_tasto_modes.png",
        "E2_ponticello.wav",
        "E2_tasto.wav",
        "E2_ponticello_then_tasto.wav",
    }

    assert style.current_theme().name == "dark"  # exports drew in light, then switched back

    compare.damping.set("flat")
    compare.sigmas[0].set("0.5")
    settle(root)
    a, b = compare.settings()
    assert (a.damping, a.sigma0, b.sigma0, a.sigma2) == ("flat", 0.5, 0.5, None)
    compare.sigmas[1].set("lots")
    compare.schedule(0)
    settle(root)
    assert "σ2" in compare.status.get()
    compare.sigmas[1].set("-1")
    compare.schedule(0)
    settle(root)
    assert "sigma2 can't be negative" in compare.status.get()
    compare.sigmas[1].set("")
    compare.damping.set("full")

    answers = []
    monkeypatch.setattr("tkinter.messagebox.askyesnocancel", lambda *a: answers.pop(0))
    app.tuner.sigma0.set_val(1.5)
    answers.append(None)  # cancel: nothing changes
    app.close_profile()
    assert app.tuner is not None and app.tuner.state.dirty
    answers.append(True)  # save, then close
    app.close_profile()  # rebuilds the tabs, keeps the session, stays on this tab
    assert app.tuner is None and not app.compare.calibrated.get()
    assert app.notebook.index("current") == list(TABS).index("explore")
    assert json.loads(config.read_text())["profile"] is None
    app.open_profile(profile_file)
    assert app.tuner is not None and app.tuner.state.damping()[0] == pytest.approx(1.5)  # saved


def test_without_a_profile_the_tuner_tab_explains(root):
    from spectrum_modeler.viz.app import App

    app = App(root, config=None, tab="tune")
    settle(root)
    assert app.tuner is None
    assert app.session_text.get() == "no session"
    assert not app.compare.calibrated.get()
    app.close()
