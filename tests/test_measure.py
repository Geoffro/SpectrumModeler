import dataclasses

import numpy as np
import pytest

from spectrum_modeler.audio import read_wav
from spectrum_modeler.measure import (
    align_model_db,
    fit_damping,
    fit_inharmonicity,
    load_measurements,
    measure,
    model_for,
    save_measurements,
)
from spectrum_modeler.modal import coefficients
from spectrum_modeler.params import ModelOptions, PluckParams
from spectrum_modeler.presets import OPEN_STRINGS, open_string
from spectrum_modeler.synthesis import render_spectrum
from spectrum_modeler.synthetic import SYMPATHETIC_RATIO, load_truth

SR = 48_000
SIGMA0, SIGMA2, B = 1.0, 2e-6, 5e-5


def synthesize(note: str, mm: float, noise_dbfs: float = -80.0, seed: int = 0):
    """A single pluck after 0.1 s of silence, with known damping and stiffness."""
    string = dataclasses.replace(open_string(note), sigma0=SIGMA0, sigma2=SIGMA2, inharmonicity_b=B)
    spectrum = coefficients(
        string,
        PluckParams(mm, width_mm=10),
        ModelOptions(n_harmonics=min(100, int(12_000 // string.f0)), inharmonicity=True),
    )
    audio = np.concatenate([np.zeros(SR // 10), 0.3 * render_spectrum(spectrum, 2.5, SR)])
    audio += np.random.default_rng(seed).normal(0, 10 ** (noise_dbfs / 20), audio.size)
    return audio, spectrum


@pytest.mark.parametrize("note", ["E2", "G3", "E4"])
@pytest.mark.parametrize("mm", [40.0, 130.0, 325.0])
def test_round_trip_recovers_known_parameters(note, mm):
    audio, spectrum = synthesize(note, mm)
    m = measure(audio, SR)  # onset and f0 both detected

    assert m.onset == pytest.approx(0.1, abs=0.001)
    assert m.f0 == pytest.approx(spectrum.string.f0, abs=0.05)
    assert m.inharmonicity_b == pytest.approx(B, rel=0.05)
    assert m.sigma0 == pytest.approx(SIGMA0, rel=0.1)
    assert m.sigma2 == pytest.approx(SIGMA2, rel=0.05)

    # Relative harmonic levels match the model.
    has_level = m.reliable & np.isfinite(m.initial_db)
    assert has_level.sum() >= 5
    model_db = 20 * np.log10(np.abs(spectrum.amps[m.harmonics[has_level] - 1]))
    deviation = m.initial_db[has_level] - model_db
    deviation -= np.median(deviation)
    assert np.percentile(np.abs(deviation), 90) < 1.5

    # Decay rates match where measured.
    k = m.harmonics[m.reliable] - 1
    rates = m.decay_rates[m.reliable]
    measured = np.isfinite(rates)
    assert measured.sum() >= 5
    np.testing.assert_allclose(rates[measured], spectrum.decay_rates[k][measured], rtol=0.15)


def test_missing_harmonics_are_not_reported_as_measured():
    audio, _ = synthesize("G3", 325.0)
    m = measure(audio, SR, f0_hint=196.0)
    even = m.harmonics % 2 == 0
    assert not m.reliable[even].any()
    assert m.reliable[~even][:10].all()


def test_f0_hint_limits_the_search():
    audio, _ = synthesize("A2", 130.0)
    m = measure(audio, SR, f0_hint=110.0)
    assert m.f0 == pytest.approx(110.0, abs=0.05)


def test_fit_helpers_recover_exact_values():
    n = np.arange(1, 30)
    freqs = n * 100.0 * np.sqrt(1 + 3e-5 * n**2)
    f0, b = fit_inharmonicity(n, freqs)
    assert f0 == pytest.approx(100.0) and b == pytest.approx(3e-5)

    rates = 0.8 + 1.5e-6 * freqs**2
    rates[5] = np.nan
    sigma0, sigma2 = fit_damping(freqs, rates)
    assert sigma0 == pytest.approx(0.8) and sigma2 == pytest.approx(1.5e-6)
    assert fit_damping(freqs[:2], rates[:2]) is None


@pytest.mark.parametrize(
    "audio",
    [np.zeros(SR), np.random.default_rng(0).normal(0, 0.05, SR)],
    ids=["silence", "noise"],
)
def test_unpitched_input_raises_a_clear_error(audio):
    with pytest.raises(ValueError, match="onset"):
        measure(audio, SR)


def test_too_short_after_onset_raises():
    audio, _ = synthesize("E2", 130.0)
    with pytest.raises(ValueError, match="Need"):
        measure(audio[: SR // 10 + SR // 20], SR, onset=SR // 10)


def test_measurements_round_trip_through_json(tmp_path):
    audio, _ = synthesize("D3", 130.0)
    m = measure(audio, SR, clip="D3_130mm", take=1, note="D3", pluck_mm=130.0, finger="flesh")
    path = tmp_path / "m" / "D3.json"
    save_measurements(path, [m])
    (loaded,) = load_measurements(path)
    for key, value in m.__dict__.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(getattr(loaded, key), value)
        else:
            assert getattr(loaded, key) == value


def test_model_overlay_needs_note_and_position():
    audio, _ = synthesize("G3", 130.0)
    bare = measure(audio, SR)
    assert model_for(bare) is None

    m = measure(audio, SR, note="G3", pluck_mm=130.0, finger="flesh")
    model = model_for(m)
    aligned = align_model_db(m, model)
    usable = m.reliable & np.isfinite(aligned) & np.isfinite(m.initial_db)
    assert np.median(np.abs(m.initial_db[usable] - aligned[usable])) < 1.0


def measure_session(session, report, file):
    clip = next(c for c in report.clips if c.file == file)
    results = []
    for i, take in enumerate(clip.takes):
        audio, sr = read_wav(session.processed / take)
        onset = round(clip.onset_in_take(i) * sr)
        results.append(measure(audio, sr, onset, f0_hint=OPEN_STRINGS[clip.note][1]))
    return clip, results


def test_sympathetic_tone_is_flagged_and_clean_clips_are_not(synthetic_session):
    session, report = synthetic_session
    truth = load_truth(session)["clips"]
    for file, clip_truth in truth.items():
        if clip_truth["impairment"] not in ("sympathetic", None):
            continue
        clip, results = measure_session(session, report, file)
        for m in results:
            if clip_truth["impairment"] == "sympathetic":
                expected = SYMPATHETIC_RATIO * OPEN_STRINGS[clip.note][1]
                assert m.off_series_hz.size == 1
                assert m.off_series_hz[0] == pytest.approx(expected, abs=2)
            else:
                assert m.off_series_hz.size == 0


def test_synthetic_session_measures_close_to_truth(synthetic_session):
    session, report = synthetic_session
    truth = load_truth(session)
    for clip in report.clips:
        if truth["clips"][clip.file]["impairment"] == "clipping":
            continue
        _, results = measure_session(session, report, clip.file)
        expected = truth["strings"][clip.note]
        assert np.median([m.inharmonicity_b for m in results]) == pytest.approx(
            expected["b"], rel=0.1
        )
        assert np.median([m.sigma0 for m in results]) == pytest.approx(expected["sigma0"], rel=0.1)
        assert np.median([m.sigma2 for m in results]) == pytest.approx(expected["sigma2"], rel=0.1)


def test_fit_damping_ignores_a_spuriously_slow_rate():
    freqs = 100.0 * np.arange(1, 21)
    rates = 0.8 + 1.5e-6 * freqs**2
    rates[2] = 0.05  # e.g. a partial fed by a sympathetic string
    sigma0, sigma2 = fit_damping(freqs, rates)
    assert sigma0 == pytest.approx(0.8, rel=1e-3) and sigma2 == pytest.approx(1.5e-6, rel=1e-3)


def test_session_without_measurements_is_a_clear_error(tmp_path):
    from spectrum_modeler.measure import load_session_measurements
    from spectrum_modeler.recording import Session

    with pytest.raises(FileNotFoundError, match=r"analyze\.py"):
        load_session_measurements(Session(tmp_path))
