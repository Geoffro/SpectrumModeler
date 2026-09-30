"""Measure one recorded pluck: f0, harmonic levels, decay rates, inharmonicity (design §4.4).

All times are relative to the pluck onset. Levels are in dBFS, where a sinusoid
of amplitude A (full scale = 1) reads 20 log10(A).
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.ndimage import maximum_filter1d

from .analysis import spectrogram
from .audio import read_wav
from .modal import ModalSpectrum, coefficients
from .params import ModelOptions, PluckParams
from .presets import FINGER_WIDTHS_MM, OPEN_STRINGS, open_string
from .recording import ClipReport, Session, detect_onsets

WINDOW_START = 0.03  # s after onset: skip the attack transient
WINDOW_LENGTH = 0.15  # s: resolves E2 harmonics, short enough that high ones are still present
# Blackman-Harris: sidelobes ~92 dB down, so a strong partial does not leak into
# the place of a missing neighbour and make it look present. Its main lobe spans
# +/- 4 bins (27 Hz at 0.15 s), still inside half the E2 harmonic spacing.
WINDOW = "blackmanharris"
MAIN_LOBE_BINS = 4
OFF_SERIES_WINDOW = 0.5  # s: longer window for sharper peaks when looking for stray tones
ZERO_PAD = 8
RELIABLE_SNR_DB = 15.0  # harmonic must be this far above the local floor
MAX_FREQ = 10_000.0  # Hz, highest harmonic measured
PITCH_RANGE = (60.0, 1000.0)  # Hz, f0 search range without a hint
HINT_CENTS = 50.0  # search +/- this around a hinted f0
MIN_DECAY_POINTS = 5
DAMP_DROP_DB = 7.5  # a broadband drop this large within 50 ms marks the note being damped
# No onset level for partials that decayed more than this by the window.
MAX_DECAY_CORRECTION_DB = 20.0


@dataclass(frozen=True, eq=False)
class WindowSpectrum:
    freqs: np.ndarray  # Hz
    db: np.ndarray  # dBFS amplitude
    times: np.ndarray  # s from onset, one per windowed sample
    weights: np.ndarray  # the window

    @property
    def df(self) -> float:
        return float(self.freqs[1] - self.freqs[0])


def window_spectrum(
    audio: np.ndarray,
    sample_rate: int,
    onset: int,
    start: float = WINDOW_START,
    length: float = WINDOW_LENGTH,
    window: str = WINDOW,
) -> WindowSpectrum:
    """Windowed, zero-padded spectrum of ``length`` s starting ``start`` s after onset."""
    i0, n = onset + round(start * sample_rate), round(length * sample_rate)
    segment = audio[i0 : i0 + n]
    if segment.size < n:
        raise ValueError(
            f"Need {start + length:.2f} s of audio after the pluck; "
            f"have {(audio.size - onset) / sample_rate:.2f} s"
        )
    weights = signal.get_window(window, n)
    n_fft = 2 ** int(np.ceil(np.log2(n))) * ZERO_PAD
    magnitude = 2 * np.abs(np.fft.rfft(segment * weights, n_fft)) / weights.sum()
    return WindowSpectrum(
        freqs=np.fft.rfftfreq(n_fft, 1 / sample_rate),
        db=20 * np.log10(magnitude + 1e-12),
        times=start + np.arange(n) / sample_rate,
        weights=weights,
    )


def decay_correction_db(ws: WindowSpectrum, sigma: float) -> float:
    """dB to add to a windowed level to get the level at the onset, for decay rate ``sigma``."""
    fraction = np.sum(ws.weights * np.exp(-sigma * ws.times)) / ws.weights.sum()
    return float(-20 * np.log10(fraction))


def _peak(ws: WindowSpectrum, f_lo: float, f_hi: float) -> tuple[float, float, bool]:
    """Highest point between ``f_lo`` and ``f_hi``, refined by parabolic interpolation.

    Also returns whether it is a true local maximum inside the range. A maximum
    at the edge is the skirt of a neighbouring peak, not a partial.
    """
    lo = max(1, int(np.ceil(f_lo / ws.df)))
    hi = min(ws.freqs.size - 2, int(np.floor(f_hi / ws.df)))
    k = lo + int(np.argmax(ws.db[lo : hi + 1]))
    a, b, c = ws.db[k - 1 : k + 2]
    interior = lo < k < hi and b >= a and b >= c
    denom = a - 2 * b + c
    p = 0.5 * (a - c) / denom if denom < 0 else 0.0
    return float(ws.freqs[k] + p * ws.df), float(b - 0.25 * (a - c) * p), interior


def _local_floor(ws: WindowSpectrum, freq: float, f0: float) -> float:
    """Median level between this partial's main lobe and half a harmonic spacing away."""
    main_lobe = MAIN_LOBE_BINS / (ws.times[-1] - ws.times[0])
    distance = np.abs(ws.freqs - freq)
    mask = (distance >= main_lobe) & (distance <= f0 / 2)
    if mask.sum() < 3:
        mask = distance <= f0 / 2
    return float(np.median(ws.db[mask]))


def estimate_f0(ws: WindowSpectrum, lo: float, hi: float, n_harmonics: int = 8) -> float:
    """Harmonic-sum pitch estimate: the f0 whose first harmonics carry the most energy.

    Each candidate scores the sum of its harmonics' heights above the median level,
    so a missing harmonic (a pluck on a node) costs a little but does not mislead.
    """
    height = np.clip(ws.db - np.median(ws.db), 0, None)
    height = maximum_filter1d(height, size=max(3, round(6.0 / ws.df)))
    candidates = np.geomspace(lo, hi, int(np.log2(hi / lo) * 240))  # 5-cent steps
    k = np.arange(1, n_harmonics + 1)
    bins = np.round(np.outer(candidates, k) / ws.df).astype(int)
    in_range = bins < height.size
    scores = np.where(in_range, height[np.minimum(bins, height.size - 1)], 0).sum(axis=1)
    if scores.max() <= 0:
        raise ValueError("No pitched sound found")
    best = candidates[np.argmax(scores)]
    return _peak(ws, best * 0.97, best * 1.03)[0]


def fit_inharmonicity(n: np.ndarray, freqs: np.ndarray) -> tuple[float, float]:
    """Fit f_n = n f0 sqrt(1 + B n^2): linear in n^2 for (f_n / n)^2. Returns (f0, B)."""
    slope, intercept = np.polyfit(n.astype(float) ** 2, (freqs / n) ** 2, 1)
    f0 = float(np.sqrt(intercept))
    return f0, float(max(slope / intercept, 0.0))


def fit_damping(freqs: np.ndarray, rates: np.ndarray) -> tuple[float, float] | None:
    """Fit sigma_n = sigma0 + sigma2 f^2 to measured rates, ignoring NaNs.

    Errors are weighted relative to the fitted rate, so the slow low partials
    (which set sigma0) count as much as the fast high ones, and rates far from
    the fit (beyond 4 robust deviations, relatively) are dropped.
    Returns (sigma0, sigma2), or None with fewer than three rates.
    """
    ok = np.isfinite(rates) & (rates > 0)
    if ok.sum() < 3:
        return None
    f2, r = freqs[ok] ** 2, rates[ok]
    sigma2, sigma0 = np.polyfit(f2, r, 1)
    keep = np.ones(r.size, dtype=bool)
    for _ in range(10):
        predicted = np.maximum(sigma0 + sigma2 * f2, 1e-3)
        relative = (r - predicted) / predicted
        spread = 1.4826 * np.median(np.abs(relative[keep])) + 1e-6
        new_keep = np.abs(relative) <= 4 * spread
        if new_keep.sum() < 3:
            break
        keep = new_keep
        sigma2, sigma0 = np.polyfit(f2[keep], r[keep], 1, w=1 / predicted[keep])
    return float(sigma0), float(sigma2)


@dataclass(frozen=True, eq=False)
class Measurement:
    clip: str
    take: int | None
    note: str | None
    pluck_mm: float | None
    finger: str | None
    sample_rate: int
    onset: float  # s into the audio that was measured
    window: tuple[float, float]  # (start, length) in s after onset
    f0: float
    inharmonicity_b: float | None
    harmonics: np.ndarray
    freqs: np.ndarray  # Hz, measured peaks
    window_db: np.ndarray  # level in the analysis window
    initial_db: np.ndarray  # extrapolated back to the onset; NaN where unreliable
    floor_db: np.ndarray  # local noise floor
    decay_rates: np.ndarray  # 1/s; NaN where not measured
    reliable: np.ndarray  # bool: at least RELIABLE_SNR_DB above the floor
    sigma0: float | None
    sigma2: float | None
    off_series_hz: np.ndarray  # strong peaks that are not harmonics

    @property
    def n_reliable(self) -> int:
        return int(self.reliable.sum())

    def to_dict(self) -> dict:
        def clean(value):
            if isinstance(value, np.ndarray):
                return [clean(v) for v in value.tolist()]
            if isinstance(value, float) and not np.isfinite(value):
                return None
            return value

        return {k: clean(v) for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, data: dict) -> "Measurement":
        arrays = {
            "harmonics": int,
            "freqs": float,
            "window_db": float,
            "initial_db": float,
            "floor_db": float,
            "decay_rates": float,
            "reliable": bool,
            "off_series_hz": float,
        }
        values = dict(data)
        for key, dtype in arrays.items():
            values[key] = np.array([np.nan if v is None else v for v in data[key]], dtype=dtype)
        values["window"] = tuple(data["window"])
        return cls(**values)


def save_measurements(path: str | Path, measurements: list[Measurement]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([m.to_dict() for m in measurements], indent=1))


def load_measurements(path: str | Path) -> list[Measurement]:
    return [Measurement.from_dict(d) for d in json.loads(Path(path).read_text())]


def measure(
    audio: np.ndarray,
    sample_rate: int,
    onset: int | None = None,
    *,
    f0_hint: float | None = None,
    clip: str = "",
    take: int | None = None,
    note: str | None = None,
    pluck_mm: float | None = None,
    finger: str | None = None,
    max_freq: float = MAX_FREQ,
) -> Measurement:
    """Measure the pluck starting at sample ``onset`` (detected if not given)."""
    audio = np.asarray(audio, dtype=float)
    if onset is None:
        onsets = detect_onsets(audio, sample_rate)
        if onsets.size == 0:
            raise ValueError("No pluck onset found")
        onset = int(onsets[0])
    ws = window_spectrum(audio, sample_rate, onset)

    if f0_hint is None:
        f0 = estimate_f0(ws, *PITCH_RANGE)
    else:
        ratio = 2 ** (HINT_CENTS / 1200)
        f0 = estimate_f0(ws, f0_hint / ratio, f0_hint * ratio)

    top = min(max_freq, 0.45 * sample_rate)
    n, freqs, levels, floors, reliable, f0, b = _track_harmonics(ws, f0, top)
    rates = _decay_rates(audio[onset:], sample_rate, freqs[reliable], f0)
    decay_rates = np.full(n.size, np.nan)
    decay_rates[reliable] = rates

    damping = fit_damping(freqs, decay_rates)
    correction_rates = np.where(
        np.isfinite(decay_rates),
        decay_rates,
        damping[0] + damping[1] * freqs**2 if damping else 0.0,
    )
    correction = np.array([decay_correction_db(ws, s) for s in correction_rates])
    initial = levels + correction
    # An extrapolated noise level means nothing, and a partial that has mostly
    # decayed by the window amplifies any error in its level by the correction.
    initial[~reliable | (correction > MAX_DECAY_CORRECTION_DB)] = np.nan

    return Measurement(
        clip=clip,
        take=take,
        note=note,
        pluck_mm=pluck_mm,
        finger=finger,
        sample_rate=sample_rate,
        onset=onset / sample_rate,
        window=(WINDOW_START, WINDOW_LENGTH),
        f0=f0,
        inharmonicity_b=b,
        harmonics=n,
        freqs=freqs,
        window_db=levels,
        initial_db=initial,
        floor_db=floors,
        decay_rates=decay_rates,
        reliable=reliable,
        sigma0=damping[0] if damping else None,
        sigma2=damping[1] if damping else None,
        off_series_hz=_off_series_peaks(audio, sample_rate, onset, f0, b or 0.0, freqs[-1]),
    )


def _track_harmonics(ws: WindowSpectrum, f0: float, top: float):
    """Find each harmonic in turn, refitting f0 and B as reliable peaks accumulate.

    Stiffness pushes upper partials sharp; predicting each from the fit so far
    keeps the search window on the right peak.
    """
    n_list, freqs, levels, floors, reliable = [], [], [], [], []
    f0_fit, b_fit = f0, 0.0
    tolerance = 1 / (ws.times[-1] - ws.times[0])  # one frequency bin of the window
    n = 1
    while True:
        expected = n * f0_fit * np.sqrt(1 + b_fit * n**2)
        if expected > top:
            break
        freq, level, interior = _peak(ws, expected - f0_fit / 4, expected + f0_fit / 4)
        floor = _local_floor(ws, expected, f0_fit)
        n_list.append(n)
        freqs.append(freq)
        levels.append(level)
        floors.append(floor)
        # A real partial is a local maximum, clear of the floor, and where the
        # fit predicts it: noise peaks wander anywhere in the search range.
        reliable.append(
            interior and level - floor >= RELIABLE_SNR_DB and abs(freq - expected) <= tolerance
        )

        good = np.array(reliable)
        if n >= 4 and good.sum() >= 3:
            f0_fit, b_fit = fit_inharmonicity(np.array(n_list)[good], np.array(freqs)[good])
        n += 1

    n_arr, good = np.array(n_list), np.array(reliable, dtype=bool)
    b = b_fit if good.sum() >= 5 else None
    return n_arr, np.array(freqs), np.array(levels), np.array(floors), good, f0_fit, b


def _decay_rates(audio: np.ndarray, sample_rate: int, freqs: np.ndarray, f0: float) -> np.ndarray:
    """Decay rate (1/s) of each partial, from a straight-line fit to its dB level over time."""
    spec = spectrogram(audio, sample_rate)
    levels = spec.db(ref=1.0, floor_db=-240)
    df = spec.freqs[1] - spec.freqs[0]

    # Stop at the note's end if it was damped: a sudden broadband drop.
    total = 10 * np.log10(np.sum(spec.magnitude**2, axis=0) + 1e-24)
    step = max(1, round(0.05 / (spec.times[1] - spec.times[0]))) if spec.times.size > 1 else 1
    drops = np.flatnonzero(
        (total[step:] - total[:-step] < -DAMP_DROP_DB) & (spec.times[:-step] > 0.1)
    )
    t_end = spec.times[drops[0]] if drops.size else spec.times[-1]

    rates = np.full(freqs.size, np.nan)
    for i, freq in enumerate(freqs):
        k = round(freq / df)
        row = levels[max(0, k - 2) : k + 3].max(axis=0)
        band = np.abs(spec.freqs - freq) <= f0 / 2
        floor = np.median(levels[band], axis=0)
        valid = (row - floor >= RELIABLE_SNR_DB) & (spec.times <= t_end)
        # the first unbroken run of valid frames from the start
        run = np.argmin(valid) if not valid.all() else valid.size
        if run >= MIN_DECAY_POINTS:
            slope = np.polyfit(spec.times[:run], row[:run], 1)[0]
            rates[i] = -slope * np.log(10) / 20
    return rates


def _off_series_peaks(
    audio: np.ndarray, sample_rate: int, onset: int, f0: float, b: float, f_max: float
) -> np.ndarray:
    """Strong, distinct peaks that do not sit on the harmonic series (e.g. sympathetic strings)."""
    available = (audio.size - onset) / sample_rate - WINDOW_START
    length = min(OFF_SERIES_WINDOW, available)
    if length < WINDOW_LENGTH:
        return np.array([])
    ws = window_spectrum(audio, sample_rate, onset, WINDOW_START, length)
    band = (ws.freqs > 40) & (ws.freqs < f_max)
    floor = np.median(ws.db[band])
    # Blackman-Harris sidelobes are ~92 dB down, so nothing within 60 dB of the
    # strongest peak is window leakage.
    height = max(floor + 20, ws.db[band].max() - 60)
    peaks, _ = signal.find_peaks(ws.db, prominence=15, height=height)
    peak_freqs = ws.freqs[peaks]
    peak_freqs = peak_freqs[(peak_freqs > 40) & (peak_freqs < f_max)]

    n = np.arange(1, int(f_max / f0) + 2)
    harmonic_freqs = n * f0 * np.sqrt(1 + b * n**2)
    tolerance = max(0.1 * f0, 12.0)
    distance = np.abs(peak_freqs[:, None] - harmonic_freqs[None, :]).min(axis=1)
    return peak_freqs[distance > tolerance]


def model_for(measurement: Measurement) -> ModalSpectrum | None:
    """The uncalibrated model for a measured open-string pluck, for comparison.

    Needs the note (one of the open strings) and pluck distance; returns None
    without them. Uses the finger preset's width, and the bridge-force spectrum.
    """
    m = measurement
    if m.note not in OPEN_STRINGS or m.pluck_mm is None:
        return None
    return coefficients(
        open_string(m.note),
        PluckParams(distance_mm=m.pluck_mm, width_mm=FINGER_WIDTHS_MM.get(m.finger or "", 0.0)),
        ModelOptions(n_harmonics=int(m.harmonics[-1]), damping=False),
    )


def align_model_db(measurement: Measurement, model: ModalSpectrum) -> np.ndarray:
    """Model levels in dB, shifted to best match the reliable measured onset levels."""
    m = measurement
    with np.errstate(divide="ignore"):
        model_db = 20 * np.log10(np.abs(model.amps[: m.harmonics.size]))
    usable = m.reliable & np.isfinite(model_db) & np.isfinite(m.initial_db)
    offset = np.median(m.initial_db[usable] - model_db[usable]) if usable.any() else 0.0
    return model_db + offset


def measure_clip(session: Session, clip: ClipReport) -> list[Measurement]:
    """Measure every take of an imported clip. Takes that can't be measured are skipped."""
    f0_hint = OPEN_STRINGS[clip.note][1] if clip.note in OPEN_STRINGS else None
    results = []
    for i, take in enumerate(clip.takes):
        audio, sr = read_wav(session.processed / take)
        try:
            results.append(
                measure(
                    audio,
                    sr,
                    round(clip.onset_in_take(i) * sr),
                    f0_hint=f0_hint,
                    clip=clip.stem,
                    take=i + 1,
                    note=clip.note,
                    pluck_mm=clip.pluck_mm,
                    finger=clip.finger,
                )
            )
        except ValueError:
            continue
    return results


def load_session_measurements(session: Session) -> list[Measurement]:
    """Every measurement saved in ``<session>/measurements/``."""
    files = sorted(session.measurements.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No measurements in {session.measurements}; run analyze.py first")
    return [m for path in files for m in load_measurements(path)]


def take_audio(session: Session, m: Measurement) -> np.ndarray:
    """The measured take's audio from its onset on."""
    audio, sr = read_wav(session.processed / f"{m.clip}_t{m.take}.wav")
    return audio[round(m.onset * sr) :]
