"""WAV reading and writing, loudness matching, and playback."""

from pathlib import Path

import numpy as np
from scipy.io import wavfile


def write_wav(
    path: str | Path, audio: np.ndarray, sample_rate: int, check_range: bool = True
) -> None:
    """Write 32-bit float WAV.

    Samples outside [-1, 1] are an error, not silently clipped, unless
    ``check_range`` is off (for storing decoded recordings as they are).
    """
    audio = np.asarray(audio)
    if check_range and np.abs(audio).max(initial=0) > 1:
        raise ValueError("Audio exceeds full scale; normalize before writing")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(path, sample_rate, audio.astype(np.float32))


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Read a WAV as float64 in [-1, 1]; shape (frames,) or (frames, channels)."""
    sample_rate, data = wavfile.read(path)
    if np.issubdtype(data.dtype, np.floating):
        return data.astype(np.float64), sample_rate
    if data.dtype == np.uint8:
        return (data.astype(np.float64) - 128) / 128, sample_rate
    return data.astype(np.float64) / -np.iinfo(data.dtype).min, sample_rate


def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio))))


def match_loudness(*signals: np.ndarray, peak: float = 0.9) -> list[np.ndarray]:
    """Scale signals to equal RMS, then share one gain so the loudest peak is ``peak``.

    Brighter sounds seem louder at equal peak level; equal RMS is a fairer A/B.
    """
    equalized = [s / rms(s) for s in signals]
    top = max(np.abs(s).max() for s in equalized)
    return [s * peak / top for s in equalized]


def sequence(signals: list[np.ndarray], sample_rate: int, gap: float = 0.5) -> np.ndarray:
    """Join signals end to end with ``gap`` seconds of silence between them."""
    silence = np.zeros(round(gap * sample_rate))
    parts = [part for s in signals for part in (s, silence)][:-1]
    return np.concatenate(parts)


def play(audio: np.ndarray, sample_rate: int, blocking: bool = True) -> None:
    """Play through the default output device (needs the sounddevice package)."""
    import sounddevice

    sounddevice.play(audio, sample_rate, blocking=blocking)
