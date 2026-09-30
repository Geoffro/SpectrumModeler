import dataclasses

import numpy as np
import pytest

from spectrum_modeler.calibrate import NODE_DEPTH_DB, calibrate, pluck_spectrum_db
from spectrum_modeler.modal import coefficients
from spectrum_modeler.params import ModelOptions
from spectrum_modeler.presets import open_string, pluck
from spectrum_modeler.synthetic import load_truth, truth_response_db


@pytest.fixture(scope="module")
def calibrated(calibration_measurements):
    session, measurements = calibration_measurements
    return session, measurements, calibrate(measurements, name="synthetic")


def true_position(truth, take):
    clip = next(v for k, v in truth["clips"].items() if k.startswith(take.clip))
    return clip["pluck_mm"][take.take - 1]


def test_recovers_response_curve_at_measured_frequencies(calibrated):
    _, _, result = calibrated
    freqs = np.unique(np.round(result.freqs, -1))
    error = result.profile.response_at(freqs) - truth_response_db(freqs)
    error -= np.median(error)  # H is only defined up to a constant
    assert np.sqrt(np.mean(error**2)) < 0.5
    assert np.percentile(np.abs(error), 95) < 1.0


def test_recovers_finger_width(calibrated):
    session, _, result = calibrated
    true_width = load_truth(session)["config"]["finger_width_mm"]
    # Weakly determined: it only moves the highest harmonics by a dB or two.
    assert result.profile.finger_width_mm["flesh"] == pytest.approx(true_width, rel=0.25)


def test_recovers_string_parameters(calibrated):
    session, _, result = calibrated
    truth = load_truth(session)["strings"]
    for note, fit in result.profile.strings.items():
        assert fit.inharmonicity_b == pytest.approx(truth[note]["b"], rel=0.05)
        assert fit.sigma0 == pytest.approx(truth[note]["sigma0"], rel=0.05)
        assert fit.sigma2 == pytest.approx(truth[note]["sigma2"], rel=0.05)
        assert fit.f0 == pytest.approx(open_string(note).f0, abs=0.05)
        assert sum(r is not None for r in fit.sigma_n) >= 5


def test_refines_pluck_positions(calibrated):
    session, _, result = calibrated
    truth = load_truth(session)
    fitted = np.array([t.fitted_mm - true_position(truth, t) for t in result.takes])
    manifest = np.array([t.manifest_mm - true_position(truth, t) for t in result.takes])
    assert np.sqrt(np.mean(fitted**2)) < np.sqrt(np.mean(manifest**2))


def test_catches_a_mis_taped_clip(calibration_measurements):
    session, measurements = calibration_measurements
    bad = "D3_130mm_flesh"
    shifted = [
        dataclasses.replace(m, pluck_mm=m.pluck_mm + 4) if m.clip == bad else m
        for m in measurements
    ]
    result = calibrate(shifted, name="x")
    truth = load_truth(session)
    takes = [t for t in result.takes if t.clip == bad]
    fitted = np.mean([abs(t.fitted_mm - true_position(truth, t)) for t in takes])
    manifest = np.mean([abs(t.manifest_mm - true_position(truth, t)) for t in takes])
    assert fitted < 0.5 * manifest


def test_most_takes_fit_within_3_db(calibrated):
    _, _, result = calibrated
    rms = np.array([t.rms_db for t in result.takes])
    assert np.mean(rms <= 3.0) >= 0.85


def test_profile_reproduces_measured_levels(calibrated):
    """The calibrated model, aligned by one gain per take, matches the measured onset levels."""
    _, measurements, result = calibrated
    errors = []
    for m in measurements:
        model = coefficients(
            open_string(m.note),
            pluck(m.pluck_mm, finger=m.finger),
            ModelOptions(n_harmonics=int(m.harmonics[-1]), profile=result.profile),
        )
        model_db = 20 * np.log10(np.abs(model.amps) + 1e-30)
        ok = np.isfinite(m.initial_db)
        diff = m.initial_db[ok] - model_db[ok]
        errors.append(np.median(np.abs(diff - np.median(diff))))
    assert np.median(errors) < 1.5


def test_pluck_spectrum_nodes_are_limited():
    n = np.arange(1, 11, dtype=float)
    levels = pluck_spectrum_db(n, np.full(10, 0.5), np.zeros(10), 0.65)
    envelope = -20 * np.log10(n)
    even = n % 2 == 0
    np.testing.assert_allclose(levels[even], envelope[even] - NODE_DEPTH_DB, atol=0.01)


def test_needs_usable_measurements(calibration_measurements):
    _, measurements = calibration_measurements
    anonymous = [dataclasses.replace(m, note=None) for m in measurements]
    with pytest.raises(ValueError, match="No measurements"):
        calibrate(anonymous, name="x")
