import time

import matplotlib.pyplot as plt
import numpy as np
import pytest

from spectrum_modeler.audio import rms
from spectrum_modeler.calibrate import calibrate
from spectrum_modeler.explorer import EDGE_MM, ExplorerState
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
    plt.close(app.fig)
