"""Shared spectral analysis: spectrograms and brightness measures."""

from dataclasses import dataclass

import numpy as np
from scipy import signal


@dataclass(frozen=True, eq=False)
class Spectrogram:
    freqs: np.ndarray  # Hz
    times: np.ndarray  # s, window centres
    magnitude: np.ndarray  # (len(freqs), len(times)), linear

    def db(self, ref: float | None = None, floor_db: float = -120.0) -> np.ndarray:
        """Magnitude in dB relative to ``ref`` (default: the loudest cell)."""
        ref = self.magnitude.max() if ref is None else ref
        with np.errstate(divide="ignore"):
            levels = 20 * np.log10(self.magnitude / ref)
        return np.maximum(levels, floor_db)


def spectrogram(
    audio: np.ndarray, sample_rate: int, window: float = 0.08, hop: float = 0.01
) -> Spectrogram:
    """Short-time Fourier transform with a Hann window of ``window`` s every ``hop`` s.

    The default 80 ms window separates the harmonics of the low E (82 Hz apart).
    Only windows lying fully inside the audio are kept, so the start and end of
    the clip do not show up as broadband clicks.
    """
    n_window = round(window * sample_rate)
    n_fft = 2 ** int(np.ceil(np.log2(n_window)) + 1)  # zero-pad x2 for smoother peaks
    stft = signal.ShortTimeFFT(
        signal.windows.hann(n_window, sym=False),
        hop=round(hop * sample_rate),
        fs=sample_rate,
        mfft=n_fft,
        scale_to="magnitude",
    )
    p0 = stft.lower_border_end[1]
    p1 = stft.upper_border_begin(len(audio))[1]
    if p1 <= p0:
        raise ValueError(f"Audio is shorter than the {window} s analysis window")
    magnitude = np.abs(stft.stft(audio, p0=p0, p1=p1))
    return Spectrogram(freqs=stft.f, times=stft.t(len(audio), p0=p0, p1=p1), magnitude=magnitude)


def spectral_centroid(freqs: np.ndarray, amps: np.ndarray) -> float:
    """Power-weighted mean frequency in Hz: a single number for brightness.

    Weighting by power rather than amplitude keeps a 1/n spectrum from being
    dominated by wherever the harmonic series happens to be cut off.
    """
    power = np.abs(amps) ** 2
    return float(np.sum(freqs * power) / np.sum(power))
