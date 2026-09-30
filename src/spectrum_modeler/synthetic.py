"""Synthetic recording sessions with known parameters (design §4.1).

Clips are rendered the way the recording guide describes them: a lead-in, then
several plucks that each ring and are damped, then silence. Every pluck passes
through a known response curve and gets its own gain and a small position error;
noise is added throughout. Optional impairments reproduce the problems the
import checks look for. The true values are saved as ``truth.json`` so the
import, measurement, and calibration code can be checked against them.
"""

import dataclasses
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import body
from .audio import write_wav
from .body import Resonance
from .modal import coefficients
from .params import ModelOptions, PluckParams
from .presets import (
    DEFAULT_SCALE_LENGTH,
    OPEN_STRING_NOTES,
    OPEN_STRINGS,
    open_string,
    pluck_position_mm,
)
from .recording import GUIDE_POSITIONS, ManifestRow, Session, write_manifest
from .synthesis import render_spectrum

TRUTH_FILE = "truth.json"
IMPAIRMENTS = ("clipping", "lossy", "missing-row", "sympathetic")

# Per-string truth: (sigma0 1/s, sigma2 1/(s Hz^2), inharmonicity B).
TRUTH_STRINGS: dict[str, tuple[float, float, float]] = {
    "E2": (0.8, 1.5e-6, 6e-5),
    "A2": (0.9, 1.6e-6, 4e-5),
    "D3": (1.0, 1.8e-6, 3e-5),
    "G3": (1.2, 2.2e-6, 2.5e-5),
    "B3": (1.3, 2.4e-6, 2e-5),
    "E4": (1.5, 2.6e-6, 1.5e-5),
}

# Truth response: body resonances (deliberately unlike the illustrative ones)
# plus a phone-like microphone roll-off below HIGHPASS_HZ.
TRUTH_RESONANCES = (
    Resonance(freq=96.0, q=6.0, gain_db=10.0),
    Resonance(freq=210.0, q=5.0, gain_db=7.0),
    Resonance(freq=380.0, q=3.0, gain_db=5.0),
    Resonance(freq=1500.0, q=1.0, gain_db=-4.0),
)
HIGHPASS_HZ = 90.0

SYMPATHETIC_RATIO = 1.5  # sympathetic tone frequency / clip f0: midway between harmonics
SYMPATHETIC_LEVEL = 0.1  # relative to the pluck's peak


def truth_response_db(freqs: np.ndarray) -> np.ndarray:
    """The known H(f) in dB: body resonances plus a 2nd-order high-pass."""
    freqs = np.asarray(freqs, dtype=float)
    ratio = (freqs / HIGHPASS_HZ) ** 2
    highpass = 20 * np.log10(ratio / np.sqrt(1 + ratio**2))
    return body.response_db(freqs, TRUTH_RESONANCES) + highpass


@dataclass(frozen=True)
class SyntheticConfig:
    notes: tuple[str, ...] = OPEN_STRING_NOTES
    positions: tuple[str | float, ...] = GUIDE_POSITIONS
    finger: str = "flesh"
    finger_width_mm: float = 11.0
    plucks: int = 3
    lead: float = 1.0  # s before the first pluck
    ring: float = 3.0  # s each pluck rings before being damped
    damp: float = 0.05  # s to damp it
    gap: float = 0.7  # s of silence after damping
    peak: float = 0.25  # peak level of a pluck before its gain
    gain_spread_db: float = 3.0  # per-pluck gain drawn from +/- this
    position_jitter_mm: float = 2.0  # true pluck point drawn from +/- this
    noise_dbfs: float = -70.0  # white noise, RMS
    scale_length: float = DEFAULT_SCALE_LENGTH  # m
    max_freq: float = 12_000.0  # highest partial rendered
    sample_rate: int = 48_000
    impairments: tuple[str, ...] = ()
    seed: int = 0

    def __post_init__(self) -> None:
        unknown = set(self.impairments) - set(IMPAIRMENTS)
        if unknown:
            raise ValueError(f"Unknown impairments {sorted(unknown)}; expected {IMPAIRMENTS}")
        if len(self.notes) * len(self.positions) <= len(self.impairments):
            raise ValueError("Need more clips than impairments (one clip per impairment)")


def generate_session(
    root: str | Path, config: SyntheticConfig | None = None, overwrite: bool = False
) -> Session:
    """Write a complete session (raw clips, manifest, truth) to ``root``."""
    config = config or SyntheticConfig()
    root = Path(root)
    if root.exists() and any(root.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{root} is not empty (use overwrite=True)")
        shutil.rmtree(root)
    session = Session(root)
    session.raw.mkdir(parents=True)

    rng = np.random.default_rng(config.seed)
    sr = config.sample_rate
    noise_rms = 10 ** (config.noise_dbfs / 20)
    rows: list[ManifestRow] = []
    truth_clips: dict[str, dict] = {}

    room = rng.normal(0, noise_rms, 5 * sr)
    write_wav(session.raw / "ROOMTONE_synthetic.wav", room, sr)
    rows.append(
        ManifestRow(file="ROOMTONE_synthetic.wav", session="synthetic", recorder="synthetic")
    )

    clip_specs = [(note, pos) for note in config.notes for pos in config.positions]
    impaired = {i + 1: name for i, name in enumerate(config.impairments)}

    for index, (note, position) in enumerate(clip_specs):
        impairment = impaired.get(index)
        distance = pluck_position_mm(position, config.scale_length)
        suffix = ".m4a" if impairment == "lossy" else ".wav"
        name = f"{note}_{round(distance):03d}mm_{config.finger}{suffix}"
        audio, clip_truth = _render_clip(note, distance, impairment, config, rng)
        audio += rng.normal(0, noise_rms, audio.size)
        if impairment == "clipping":
            audio = np.clip(4 * audio, -1, 1)

        if suffix == ".m4a":
            _write_aac(session.raw / name, audio, sr)
        else:
            write_wav(session.raw / name, audio, sr)
        if impairment != "missing-row":
            rows.append(
                ManifestRow(
                    file=name,
                    string=OPEN_STRINGS[note][0],
                    note=note,
                    pluck_mm=distance,
                    finger=config.finger,
                    stroke="free",
                    plucks=config.plucks,
                    session="synthetic",
                    recorder="synthetic",
                    mic_distance_cm=40.0,
                )
            )
        truth_clips[name] = {**clip_truth, "impairment": impairment}

    write_manifest(session.manifest, rows)
    truth = {
        "config": dataclasses.asdict(config),
        "strings": {
            n: dict(zip(("sigma0", "sigma2", "b"), TRUTH_STRINGS[n], strict=True))
            for n in config.notes
        },
        "response": {
            "resonances": [dataclasses.asdict(r) for r in TRUTH_RESONANCES],
            "highpass_hz": HIGHPASS_HZ,
        },
        "clips": truth_clips,
    }
    (root / TRUTH_FILE).write_text(json.dumps(truth, indent=2))
    return session


def load_truth(session: Session) -> dict:
    return json.loads((session.root / TRUTH_FILE).read_text())


def _render_clip(
    note: str,
    distance_mm: float,
    impairment: str | None,
    config: SyntheticConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict]:
    sr = config.sample_rate
    sigma0, sigma2, b = TRUTH_STRINGS[note]
    string = dataclasses.replace(
        open_string(note, length=config.scale_length),
        sigma0=sigma0,
        sigma2=sigma2,
        inharmonicity_b=b,
    )
    opts = ModelOptions(
        n_harmonics=min(100, int(config.max_freq // string.f0)),
        inharmonicity=True,
    )
    ring = round(config.ring * sr)
    damp = round(config.damp * sr)
    fade_out = np.ones(ring)
    fade_out[-damp:] = 0.5 + 0.5 * np.cos(np.pi * np.arange(damp) / damp)
    t = np.arange(ring) / sr

    parts = [np.zeros(round(config.lead * sr))]
    onsets, gains, true_mm = [], [], []
    position = parts[0].size
    for _ in range(config.plucks):
        mm = distance_mm + rng.uniform(-1, 1) * config.position_jitter_mm
        gain_db = rng.uniform(-1, 1) * config.gain_spread_db
        spectrum = coefficients(
            string, PluckParams(distance_mm=mm, width_mm=config.finger_width_mm), opts
        )
        spectrum = dataclasses.replace(
            spectrum, amps=spectrum.amps * 10 ** (truth_response_db(spectrum.freqs) / 20)
        )
        pluck = render_spectrum(spectrum, config.ring, sr) * config.peak * 10 ** (gain_db / 20)
        if impairment == "sympathetic":
            f_sym = SYMPATHETIC_RATIO * string.f0
            pluck += SYMPATHETIC_LEVEL * config.peak * np.exp(-t) * np.sin(2 * np.pi * f_sym * t)
        parts += [pluck * fade_out, np.zeros(round(config.gap * sr))]
        onsets.append(position / sr)
        gains.append(gain_db)
        true_mm.append(mm)
        position += ring + parts[-1].size

    truth = {"note": note, "onsets_s": onsets, "gains_db": gains, "pluck_mm": true_mm}
    return np.concatenate(parts), truth


def _write_aac(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-f", "f32le", "-ar", str(sample_rate), "-ac", "1", "-i", "-",
            "-c:a", "aac", "-b:a", "128k", str(path),
        ],
        input=audio.astype("<f4").tobytes(),
        check=True,
        capture_output=True,
    )  # fmt: skip
