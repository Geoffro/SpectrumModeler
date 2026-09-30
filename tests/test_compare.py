import matplotlib.pyplot as plt
import numpy as np
import pytest

from spectrum_modeler.audio import rms, write_wav
from spectrum_modeler.compare import compare_audio, compare_files, harmonic_track, pluck_segment
from spectrum_modeler.presets import open_string, pluck
from spectrum_modeler.synthesis import render

SR = 48_000
E2 = open_string("E2")


def pluck_audio(position, level=0.3, lead=0.2, duration=1.5, seed=0):
    note = level * render(E2, pluck(position, finger="flesh"), duration=duration, sample_rate=SR)
    audio = np.concatenate([np.zeros(round(lead * SR)), note])
    return audio + np.random.default_rng(seed).normal(0, 1e-4, audio.size)


@pytest.fixture(scope="module")
def comparison():
    # A over the fingerboard, B near the bridge and much quieter: loudness matching evens it out.
    a = pluck_segment(pluck_audio("tasto"), SR)
    b = pluck_segment(pluck_audio("ponticello", level=0.05, seed=1), SR)
    return compare_audio(a, b, SR, names=("tasto", "ponticello"), f0_hints=(E2.f0, E2.f0))


def test_aligned_trimmed_and_loudness_matched(comparison):
    assert comparison.a.audio.size == comparison.b.audio.size
    assert rms(comparison.a.audio) == pytest.approx(rms(comparison.b.audio))
    assert comparison.a.measurement.onset == 0.0  # each starts at its pluck


def test_difference_shows_the_brighter_pluck(comparison):
    difference = comparison.difference_db()
    spec = comparison.a.spectrogram
    early = spec.times < 0.3
    upper = (spec.freqs > 1000) & (spec.freqs < 3000)
    assert np.nanmedian(difference[np.ix_(upper, early)]) > 5  # B (ponticello) brighter


def test_noise_is_blanked_in_the_difference(comparison):
    difference = comparison.difference_db()
    spec = comparison.a.spectrogram
    very_high = spec.freqs > 15_000  # nothing but noise up here
    assert np.isnan(difference[very_high]).mean() > 0.99


def test_harmonic_tracks_decay(comparison):
    track = harmonic_track(comparison, comparison.a, 1)
    times = comparison.a.spectrogram.times
    assert track[times < 0.2].mean() > track[(times > 1.0) & (times < 1.2)].mean()
    assert harmonic_track(comparison, comparison.a, 500) is None


def test_pluck_selection(tmp_path):
    two = np.concatenate(
        [pluck_audio("tasto", duration=1.0), pluck_audio("ponticello", duration=1.0)]
    )
    second = pluck_segment(two, SR, pluck=2)
    assert second.size == pytest.approx(1.2 * SR, abs=0.25 * SR)  # starts at the second onset
    with pytest.raises(ValueError, match="only 2"):
        pluck_segment(two, SR, pluck=3)
    with pytest.raises(ValueError, match="no pluck"):
        pluck_segment(np.zeros(SR), SR)


def test_files_of_any_length_and_the_plot(tmp_path):
    from spectrum_modeler.viz.compare import plot_comparison

    write_wav(tmp_path / "a.wav", pluck_audio("normal", duration=2.0), SR)
    write_wav(tmp_path / "b.wav", pluck_audio("twelfth_fret", duration=1.0), SR)
    result = compare_files(tmp_path / "a.wav", tmp_path / "b.wav", f0_hints=(E2.f0, E2.f0))
    assert result.a.name == "a" and result.duration < 1.1
    fig = plot_comparison(result)
    assert len([ax for ax in fig.axes if ax.get_visible()]) == 10  # 8 panels + 2 colorbars
    plt.close(fig)


def test_unmeasurable_recording_is_reported_not_fatal():
    a = pluck_segment(pluck_audio("normal"), SR)
    click = np.zeros(SR)
    click[100:110] = 0.5  # a click: has an onset, but no pitch
    result = compare_audio(a, click[100:], SR)
    assert result.a.measurement is not None
    assert result.b.measurement is None and result.b.problem
