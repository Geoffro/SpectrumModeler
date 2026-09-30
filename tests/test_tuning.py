import matplotlib.pyplot as plt
import numpy as np
import pytest

from spectrum_modeler.audio import rms
from spectrum_modeler.calibrate import calibrate
from spectrum_modeler.profile import load_profile
from spectrum_modeler.tuning import TunerState


@pytest.fixture
def state(calibration_measurements, tmp_path):
    session, measurements = calibration_measurements
    profile = calibrate(measurements, name="synthetic").profile
    path = tmp_path / "synthetic.json"
    profile.save(path)
    return TunerState(load_profile(path), measurements, path, session)


def test_adjustments_go_to_manual_and_survive_save(state):
    note = state.note
    fitted = state.profile.strings[note].sigma0
    state.set_damping(sigma0=fitted + 0.5)
    state.set_finger_width(6.0)
    state.set_inharmonicity(1e-4)
    assert state.dirty

    state.save()
    assert not state.dirty
    reloaded = load_profile(state.path)
    assert reloaded.strings[note].sigma0 == fitted  # fitted value untouched
    assert reloaded.damping(note)[0] == pytest.approx(fitted + 0.5)
    assert reloaded.finger_width(state.current.finger) == 6.0
    assert reloaded.inharmonicity(note) == 1e-4


def test_match_harmonic_sets_the_offset_that_closes_the_gap(state):
    harmonics, measured, model, reliable = state.levels()
    usable = np.flatnonzero(reliable & np.isfinite(measured) & np.isfinite(model))
    # choose the harmonic with the largest misfit
    i = usable[np.argmax(np.abs(measured[usable] - model[usable]))]
    before = measured[i] - model[i]
    offset = state.match_harmonic(int(harmonics[i]))
    assert offset == pytest.approx(before, abs=0.1)
    # the model is re-aligned after the change, so allow for the shift in the take's gain
    _, measured, model, _ = state.levels()
    assert abs(measured[i] - model[i]) < abs(before)


def test_zero_offset_removes_the_entry_and_clear_resets(state):
    state.set_offset(5, 3.0)
    assert state.offset(5) == 3.0
    state.set_offset(5, 0.0)
    assert 5 not in state.profile.manual[state.note].offsets_db
    state.set_damping(sigma0=3.0)
    state.clear_string()
    assert state.note not in state.profile.manual


def test_selection_wraps_and_resets_view_shift(state):
    state.position_shift_mm = 2.0
    state.step(-1)
    assert state.index == len(state.takes) - 1
    assert state.position_shift_mm == 0.0


def test_ab_audio_is_matched_in_loudness(state):
    recording, model, sr = state.ab_audio()
    assert sr == 48_000
    assert recording.size == model.size
    assert rms(recording) == pytest.approx(rms(model))


def test_tuner_window_builds_and_responds(state):
    from spectrum_modeler.viz.tuner import TunerApp

    app = TunerApp(state)
    app.sigma0.set_val(2.5)
    assert state.damping()[0] == pytest.approx(2.5)
    app._select(1)
    assert state.index == 1
    assert not state.dirty or state.note in state.profile.manual
    app._save()
    assert "saved" in app.status.get_text()
    plt.close(app.fig)


def test_model_uses_the_profiles_string_set(state, tmp_path):
    notes = sorted(state.profile.strings)
    path = tmp_path / "custom.toml"
    path.write_text("".join(f"[strings.{n}]\ntension_n = 50.0\n" for n in notes))
    state.profile.string_set = str(path)
    assert state.model().string.tension == 50.0
