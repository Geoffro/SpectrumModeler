import copy
import time

import matplotlib.pyplot as plt
import numpy as np
import pytest

from spectrum_modeler.audio import rms
from spectrum_modeler.calibrate import calibrate
from spectrum_modeler.explorer import EDGE_MM, ExplorerState, Settings
from spectrum_modeler.presets import OPEN_STRING_NOTES
from spectrum_modeler.recording import SAMPLE_RATE


@pytest.fixture(scope="module")
def profile(calibration_measurements):
    return calibrate(calibration_measurements[1], name="synthetic").profile


def test_uncalibrated_every_string_and_position():
    state = ExplorerState()
    assert not state.settings.calibrated
    for note in OPEN_STRING_NOTES:
        state.update(note=note)
        for mm in (20, 130, 325, 600):
            state.update(position_mm=mm)
            spectrum = state.spectrum()
            assert spectrum.string.note == note
            assert np.all(np.isfinite(spectrum.amps))


def test_brighter_near_the_bridge():
    state = ExplorerState()
    state.update(position_mm=40)
    near = state.centroid()
    state.update(position_mm=250)
    assert near > 1.5 * state.centroid()


def test_position_is_kept_on_the_string():
    state = ExplorerState()
    state.update(position_mm=0)
    assert state.settings.position_mm == EDGE_MM
    state.update(position_mm=10_000)
    assert state.settings.position_mm == state.max_position_mm


def test_calibrated_mode_uses_the_profile(profile):
    state = ExplorerState(profile=profile)
    s = state.settings
    assert s.calibrated and s.inharmonicity and s.body  # calibrating switches both on
    calibrated = state.spectrum()
    assert calibrated.string.f0 == profile.strings[s.note].f0
    assert calibrated.pluck.width_mm == pytest.approx(profile.finger_width("flesh"))

    state.update(calibrated=False, inharmonicity=False, body=False)
    assert not np.allclose(state.spectrum().amps, calibrated.amps)


def test_calibrating_without_a_profile_is_refused():
    with pytest.raises(ValueError, match="profile"):
        ExplorerState().update(calibrated=True)


def test_missing_stiffness_is_explained():
    state = ExplorerState()  # EJ46 has no diameters, so no B without calibration
    state.update(inharmonicity=True)
    spectrum = state.spectrum()
    np.testing.assert_allclose(spectrum.freqs, spectrum.harmonics * spectrum.string.f0)
    assert any("stiffness" in notice for notice in state.notices())


def test_damping_modes_and_overrides():
    state = ExplorerState()
    full = state.ring_times()
    assert full[1] > 5 * full[40]  # high harmonics die first

    state.update(damping="flat")
    flat = state.ring_times()
    assert flat[1] == pytest.approx(flat[40]) == pytest.approx(np.log(1000) / 1.0)
    assert "flat damping" in state.settings.describe()

    state.update(damping="full", sigma2=0.5e-6)
    assert state.damping_values() == (1.0, 0.5e-6)
    assert "σ2 5.0e-07" in state.settings.describe()
    assert state.ring_times()[40] > 2 * full[40]
    state.update(note="A2")  # overrides are kept from string to string
    assert state.damping_values()[1] == 0.5e-6

    state.update(damping="off")
    assert "forever" in "\n".join(state.readout())
    state.update(damping="flat", sigma0=0.0)
    assert "h1 forever" in "\n".join(state.readout())

    with pytest.raises(ValueError, match="damping"):
        state.update(damping="heavy")
    with pytest.raises(ValueError, match="sigma0"):
        state.update(sigma0=-1)
    with pytest.raises(ValueError, match="sigma2"):
        Settings(sigma2=-1e-6)  # as the Compare tab builds them


def test_damping_is_audible():
    state = ExplorerState()
    state.update(damping="off")
    undamped = state.audio(duration=2.0)
    state.update(damping="full")
    damped = state.audio(duration=2.0)
    late = slice(SAMPLE_RATE, 2 * SAMPLE_RATE)
    assert rms(damped[late]) < 0.5 * rms(undamped[late])


def test_calibrated_ring_times_are_measured_until_adjusted(profile):
    state = ExplorerState(profile=profile)
    assert "(measured)" in "\n".join(state.readout())
    fitted = profile.damping(state.settings.note)
    assert state.damping_values() == pytest.approx(fitted)
    state.update(sigma2=fitted[1])
    assert "(measured)" not in "\n".join(state.readout())

    tuned = copy.deepcopy(profile)  # a manual adjustment in the tuner also uses the formula
    tuned.manual_for(state.settings.note).sigma0 = fitted[0]
    assert "(measured)" not in "\n".join(ExplorerState(profile=tuned).readout())


def test_pin_and_ab_audio_are_loudness_matched():
    state = ExplorerState()
    with pytest.raises(ValueError, match="Pin"):
        state.ab_audio()
    state.update(position_mm=40)
    state.pin()
    state.update(position_mm=250)
    audio = state.ab_audio(duration=1.0, gap=0.5)
    n = SAMPLE_RATE
    a, b = audio[:n], audio[n + n // 2 :]
    assert rms(a) == pytest.approx(rms(b))
    assert np.abs(audio).max() == pytest.approx(0.9)
    assert "B: E2, 40 mm" in "\n".join(state.readout())


def test_nearest_recording(calibration_measurements):
    session, measurements = calibration_measurements
    state = ExplorerState(session=session, measurements=measurements)
    state.update(note="D3", position_mm=210)
    nearest = state.nearest_recording()
    assert nearest.note == "D3" and nearest.pluck_mm == pytest.approx(200)
    assert state.recording_audio().size > 0
    state.update(note="G3")  # not in this session
    assert state.nearest_recording() is None
    with pytest.raises(ValueError, match="No recording"):
        state.recording_audio()


def test_window_builds_updates_quickly_and_pins(profile):
    from spectrum_modeler.viz.explorer import ExplorerApp

    app = ExplorerApp(ExplorerState(profile=profile))
    canvas = app.fig.canvas
    canvas.draw_idle = lambda *args, **kwargs: None  # a GUI backend coalesces these

    before = [bar.get_height() for bar in app.bars]
    start = time.perf_counter()
    for mm in range(40, 240, 20):
        app.position.set_val(mm)
    per_update = (time.perf_counter() - start) / 10
    assert per_update < 0.05
    assert [bar.get_height() for bar in app.bars] != before

    app._pin()
    assert app.pinned_dots.get_xdata().size == 40
    app.strings.set_active(OPEN_STRING_NOTES.index("E4"))
    assert app.state.settings.note == "E4"
    app.calibrated.set_active(0)  # untick calibration
    assert not app.state.settings.calibrated

    app.damping.set_active(1)
    assert app.state.settings.damping == "flat"
    assert app.sigma2.poly.get_alpha() < 1 and app.sigma0.poly.get_alpha() == 1  # flat ignores σ2
    app.sigma0.set_val(2.0)
    app.sigma2.set_val(0.5)
    assert (app.state.settings.sigma0, app.state.settings.sigma2) == (2.0, pytest.approx(5e-7))
    app.reset_damping._observers.process("clicked", None)
    assert app.state.settings.sigma0 is None
    assert app.sigma0.val == pytest.approx(1.0) and app.sigma2.val == pytest.approx(2.0)
    plt.close(app.fig)


def test_window_sliders_follow_the_calibrated_string(profile):
    from spectrum_modeler.viz.explorer import ExplorerApp

    app = ExplorerApp(ExplorerState(profile=profile))
    app.fig.canvas.draw_idle = lambda *args, **kwargs: None
    for note in profile.strings:
        app.strings.set_active(OPEN_STRING_NOTES.index(note))
        sigma0, sigma2 = profile.damping(note)
        assert app.sigma0.val == pytest.approx(sigma0)
        assert app.sigma2.val == pytest.approx(sigma2 * 1e6)
        assert app.state.settings.sigma0 is None  # showing a value doesn't override it
    plt.close(app.fig)
