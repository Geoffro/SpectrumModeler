import dataclasses

import numpy as np
import pytest
from scipy.signal import hilbert

from spectrum_modeler.analysis import spectral_centroid, spectrogram
from spectrum_modeler.audio import match_loudness, read_wav, rms, sequence, write_wav
from spectrum_modeler.modal import coefficients
from spectrum_modeler.params import ModelOptions
from spectrum_modeler.presets import open_string, pluck
from spectrum_modeler.synthesis import render, render_spectrum

SR = 48_000
E2 = open_string("E2")
G3 = open_string("G3")


def test_render_is_finite_and_peak_normalized():
    audio = render(E2, pluck("ponticello", finger="flesh"), duration=1.0, sample_rate=SR)
    assert audio.shape == (SR,)
    assert np.all(np.isfinite(audio))
    assert np.abs(audio).max() == pytest.approx(1.0)


def test_unnormalized_render_starts_at_bridge_force_at_release():
    spectrum = coefficients(E2, pluck("normal"))
    audio = render_spectrum(spectrum, duration=0.01, sample_rate=SR, normalize=False, fade_in=0)
    assert audio[0] == pytest.approx(spectrum.amps.sum())


def test_fade_in_smooths_only_the_initial_step():
    opts = ModelOptions(n_harmonics=40)
    unfaded = render(E2, pluck("ponticello"), opts, 0.1, SR, normalize=False, fade_in=0)
    faded = render(E2, pluck("ponticello"), opts, 0.1, SR, normalize=False)
    # All modes are in phase at release, so the unfaded audio starts near its peak.
    assert abs(unfaded[0]) > 0.5 * np.abs(unfaded).max()
    assert faded[0] == 0
    # The fade lasts one cycle of the highest partial (40 x 82 Hz: ~0.3 ms), no longer.
    n_fade = round(SR / (40 * E2.f0))
    np.testing.assert_array_equal(faded[n_fade:], unfaded[n_fade:])


def test_single_partial_decays_at_sigma():
    opts = ModelOptions(n_harmonics=1)
    audio = render(G3, pluck("normal"), opts, duration=2.0, sample_rate=SR, fade_in=0)
    envelope = np.abs(hilbert(audio))
    t = np.arange(audio.size) / SR
    middle = (t > 0.2) & (t < 1.8)  # avoid Hilbert edge effects
    slope = np.polyfit(t[middle], np.log(envelope[middle]), 1)[0]
    assert -slope == pytest.approx(G3.sigma0 + G3.sigma2 * G3.f0**2, rel=1e-3)


def test_high_harmonics_decay_faster_in_the_spectrogram():
    audio = render(G3, pluck("ponticello"), duration=2.0, sample_rate=SR)
    spec = spectrogram(audio, SR)
    levels = spec.db()

    def decay_db_per_s(freq):
        row = levels[np.argmin(np.abs(spec.freqs - freq))]
        span = (spec.times > 0.2) & (spec.times < 1.0)
        return -np.polyfit(spec.times[span], row[span], 1)[0]

    assert decay_db_per_s(10 * G3.f0) > 2 * decay_db_per_s(G3.f0)


def test_partials_above_nyquist_are_dropped():
    opts = ModelOptions(n_harmonics=40, damping=False)
    audio = render(G3, pluck("ponticello"), opts, duration=1.0, sample_rate=8000)
    spec = spectrogram(audio, 8000)
    # 196 Hz x 40 = 7.8 kHz would alias below 4 kHz; only harmonics below 4 kHz may appear
    mean_spectrum = spec.magnitude.mean(axis=1)
    peak_freq = spec.freqs[np.argmax(mean_spectrum)]
    assert peak_freq == pytest.approx(G3.f0, abs=10)
    assert np.all(np.isfinite(audio))


def test_ponticello_is_brighter_than_tasto():
    centroids = {
        name: spectral_centroid(s.freqs, s.amps)
        for name in ("ponticello", "tasto")
        for s in [coefficients(E2, pluck(name, finger="flesh"))]
    }
    assert centroids["ponticello"] > 1.2 * centroids["tasto"]


def test_spectral_centroid_of_known_spectrum():
    assert spectral_centroid(np.array([100.0, 300.0]), np.array([1.0, -1.0])) == 200.0


def test_match_loudness_equalizes_rms_under_a_shared_peak():
    quiet = 0.1 * np.sin(np.linspace(0, 100, 4800))
    spiky = np.zeros(4800)
    spiky[::100] = 1.0
    a, b = match_loudness(quiet, spiky, peak=0.9)
    assert rms(a) == pytest.approx(rms(b))
    assert max(np.abs(a).max(), np.abs(b).max()) == pytest.approx(0.9)


def test_sequence_inserts_silence():
    joined = sequence([np.ones(10), np.ones(20)], sample_rate=100, gap=0.5)
    assert joined.size == 10 + 50 + 20
    assert np.all(joined[10:60] == 0)


def test_wav_round_trip(tmp_path):
    audio = render(E2, pluck("normal"), duration=0.2, sample_rate=SR)
    path = tmp_path / "sub" / "note.wav"
    write_wav(path, audio, SR)
    loaded, sr = read_wav(path)
    assert sr == SR
    np.testing.assert_allclose(loaded, audio, atol=1e-7)


def test_write_wav_refuses_to_clip(tmp_path):
    with pytest.raises(ValueError, match="full scale"):
        write_wav(tmp_path / "loud.wav", np.array([0.5, 1.5]), SR)


def test_read_wav_scales_integer_formats(tmp_path):
    from scipy.io import wavfile

    path = tmp_path / "int16.wav"
    wavfile.write(path, SR, np.array([0, 16384, -32768], dtype=np.int16))
    loaded, _ = read_wav(path)
    np.testing.assert_allclose(loaded, [0, 0.5, -1.0])


def test_layers_change_the_audio():
    base = ModelOptions()
    p = pluck("ponticello", finger="flesh")
    stiff = dataclasses.replace(E2, inharmonicity_b=1e-4)
    reference = render(stiff, p, base, duration=0.5, sample_rate=SR)
    for change in (
        {"damping": False},
        {"finger_width": False},
        {"inharmonicity": True},
        {"body": True},
    ):
        varied = render(stiff, p, dataclasses.replace(base, **change), duration=0.5, sample_rate=SR)
        assert np.abs(varied - reference).max() > 1e-3, change
