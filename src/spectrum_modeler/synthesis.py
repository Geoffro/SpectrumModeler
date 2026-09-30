"""Additive synthesis: sum each mode as an exponentially decaying cosine."""

import numpy as np

from .modal import ModalSpectrum, coefficients
from .params import ModelOptions, PluckParams, StringParams

DEFAULT_SAMPLE_RATE = 48_000

# All modes are in phase at release, so the sum starts near its peak: a step up
# from silence. Every later edge of the waveform is band-limited by the number of
# partials, but that first step is not. By default it is smoothed over one cycle
# of the highest partial, to the same bandwidth. A longer fade would blur the
# first pulse, which for a pluck near the bridge lasts well under a millisecond.


def render_spectrum(
    spectrum: ModalSpectrum,
    duration: float = 4.0,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    normalize: bool = True,
    fade_in: float | None = None,
) -> np.ndarray:
    """Audio for a computed spectrum, starting at the moment of release.

    Partials at or above the Nyquist frequency are left out. ``fade_in`` seconds
    of raised-cosine fade are applied at the start (default: one cycle of the
    highest partial). With ``normalize``, the peak is scaled to 1.0; otherwise
    samples are in the spectrum's units.
    """
    t = np.arange(round(duration * sample_rate)) / sample_rate
    audio = np.zeros_like(t)
    audible = spectrum.freqs < sample_rate / 2
    for freq, amp, decay in zip(
        spectrum.freqs[audible],
        spectrum.amps[audible],
        spectrum.decay_rates[audible],
        strict=True,
    ):
        if amp != 0:
            audio += amp * np.exp(-decay * t) * np.cos(2 * np.pi * freq * t)

    if fade_in is None:
        fade_in = 1 / spectrum.freqs[audible].max() if audible.any() else 0.0
    n_fade = min(round(fade_in * sample_rate), len(audio))
    if n_fade:
        audio[:n_fade] *= 0.5 - 0.5 * np.cos(np.pi * np.arange(n_fade) / n_fade)

    if normalize:
        peak = np.abs(audio).max()
        if peak > 0:
            audio /= peak
    return audio


def render(
    string: StringParams,
    pluck: PluckParams,
    opts: ModelOptions | None = None,
    duration: float = 4.0,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    normalize: bool = True,
    fade_in: float | None = None,
) -> np.ndarray:
    """Audio for one pluck of one string."""
    spectrum = coefficients(string, pluck, opts)
    return render_spectrum(spectrum, duration, sample_rate, normalize, fade_in)
