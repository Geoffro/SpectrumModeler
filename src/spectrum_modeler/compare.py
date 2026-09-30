"""Compare two recordings: aligned at their plucks, matched in loudness, measured alike.

Each recording starts at its chosen pluck and runs until its next pluck (or the
end); both are trimmed to the shorter length, then matched in loudness (RMS), so
differences in the spectrograms are differences in spectral balance and decay,
not in how hard the string was plucked or how far the phone was.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .analysis import Spectrogram, spectrogram
from .audio import match_loudness
from .measure import Measurement, measure
from .recording import SAMPLE_RATE, decode, detect_onsets

MAX_DURATION = 4.0  # s compared, at most
FLOOR_DB = -80.0  # spectrogram floor, relative to the louder recording's peak
VISIBLE_MARGIN_DB = 15.0  # difference shown only where a recording is this far above its noise


@dataclass(frozen=True, eq=False)
class Recording:
    name: str
    audio: np.ndarray  # from the pluck onset, loudness-matched
    spectrogram: Spectrogram
    measurement: Measurement | None  # None if it could not be measured
    problem: str | None = None  # why not


@dataclass(frozen=True, eq=False)
class Comparison:
    a: Recording
    b: Recording
    sample_rate: int
    ref: float  # 0 dB for both spectrograms

    def levels_db(self, which: Recording) -> np.ndarray:
        return which.spectrogram.db(ref=self.ref, floor_db=FLOOR_DB)

    def noise_floor_db(self, which: Recording) -> float:
        """Typical level of a spectrogram cell: most cells hold only noise."""
        return float(np.median(self.levels_db(which)))

    def difference_db(self, margin_db: float = VISIBLE_MARGIN_DB) -> np.ndarray:
        """B minus A in dB; NaN where neither recording is ``margin_db`` above its own noise.

        Each recording's own floor matters: loudness matching raises the quieter
        recording's noise along with its signal.
        """
        a, b = self.levels_db(self.a), self.levels_db(self.b)
        visible = (a > self.noise_floor_db(self.a) + margin_db) | (
            b > self.noise_floor_db(self.b) + margin_db
        )
        difference = b - a
        difference[~visible] = np.nan
        return difference

    @property
    def duration(self) -> float:
        return self.a.audio.size / self.sample_rate


def pluck_segment(audio: np.ndarray, sample_rate: int, pluck: int = 1) -> np.ndarray:
    """The audio from pluck number ``pluck`` (1-based) up to the next one."""
    onsets = detect_onsets(audio, sample_rate)
    if onsets.size == 0:
        raise ValueError("no pluck found")
    if not 1 <= pluck <= onsets.size:
        raise ValueError(f"pluck {pluck} requested, but only {onsets.size} found")
    end = onsets[pluck] if pluck < onsets.size else audio.size
    return audio[onsets[pluck - 1] : end]


def compare_audio(
    a: np.ndarray,
    b: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    *,
    names: tuple[str, str] = ("A", "B"),
    f0_hints: tuple[float | None, float | None] = (None, None),
    max_duration: float = MAX_DURATION,
) -> Comparison:
    """Compare two plucks, each starting at its onset."""
    n = min(a.size, b.size, round(max_duration * sample_rate))
    a, b = match_loudness(a[:n], b[:n])
    spectrograms = spectrogram(a, sample_rate), spectrogram(b, sample_rate)
    ref = max(s.magnitude.max() for s in spectrograms)

    recordings = []
    for audio, name, spec, hint in zip((a, b), names, spectrograms, f0_hints, strict=True):
        try:
            m = measure(audio, sample_rate, 0, f0_hint=hint, clip=name)
            recordings.append(Recording(name, audio, spec, m))
        except ValueError as e:
            recordings.append(Recording(name, audio, spec, None, str(e)))
    return Comparison(recordings[0], recordings[1], sample_rate, ref)


def compare_files(
    a: str | Path,
    b: str | Path,
    *,
    plucks: tuple[int, int] = (1, 1),
    names: tuple[str, str] | None = None,
    f0_hints: tuple[float | None, float | None] = (None, None),
    max_duration: float = MAX_DURATION,
) -> Comparison:
    """Compare pluck ``plucks[i]`` of each file (any format ffmpeg reads)."""
    segments = []
    for path, pluck in zip((a, b), plucks, strict=True):
        try:
            segments.append(pluck_segment(decode(path), SAMPLE_RATE, pluck))
        except ValueError as e:
            raise ValueError(f"{Path(path).name}: {e}") from None
    return compare_audio(
        *segments,
        names=names or (Path(a).stem, Path(b).stem),
        f0_hints=f0_hints,
        max_duration=max_duration,
    )


def harmonic_track(comparison: Comparison, which: Recording, harmonic: int) -> np.ndarray | None:
    """Level (dB, shared reference) of one harmonic over time, or None if not measured."""
    m = which.measurement
    if m is None or harmonic > m.harmonics.size or not m.reliable[harmonic - 1]:
        return None
    spec = which.spectrogram
    k = round(m.freqs[harmonic - 1] / (spec.freqs[1] - spec.freqs[0]))
    return comparison.levels_db(which)[max(0, k - 2) : k + 3].max(axis=0)
