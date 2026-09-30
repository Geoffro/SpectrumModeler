"""Guitar profiles: calibrated values for one guitar as recorded (design §4.7).

Every value is kept twice: as fitted by calibration and as adjusted by hand in
the tuner. The effective value is the manual one if set, otherwise the fitted
one. A refit replaces only fitted values, so manual adjustments survive it.
"""

import copy
import dataclasses
import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from .params import StringParams

PROFILE_DIR = Path("calibration")


@dataclass
class StringFit:
    """Fitted values for one string."""

    f0: float  # Hz, as tuned when recorded
    inharmonicity_b: float | None
    sigma0: float  # 1/s
    sigma2: float  # 1/(s Hz^2)
    # Median measured decay rate for each harmonic (index 0 = harmonic 1); None
    # where never measured. Used in place of the formula where available.
    sigma_n: list[float | None] = field(default_factory=list)


@dataclass
class StringManual:
    """Hand adjustments for one string; None means use the fitted value."""

    sigma0: float | None = None
    sigma2: float | None = None
    inharmonicity_b: float | None = None
    offsets_db: dict[int, float] = field(default_factory=dict)  # harmonic -> dB

    def is_empty(self) -> bool:
        return self == StringManual()

    def describe(self) -> str:
        parts = [f"{k}={getattr(self, k):g}" for k in ("sigma0", "sigma2", "inharmonicity_b")
                 if getattr(self, k) is not None]  # fmt: skip
        if self.offsets_db:
            parts.append("offsets on harmonics " + ", ".join(map(str, sorted(self.offsets_db))))
        return "; ".join(parts)


@dataclass
class GuitarProfile:
    name: str
    scale_length_mm: float
    string_set: str
    recorder: str | None
    created: str  # ISO date
    source_measurements: list[str]
    response_freqs: list[float]  # Hz, log-spaced
    response_db: list[float]  # H(f): body, room, and microphone
    finger_width_mm: dict[str, float]  # fitted, per finger type
    strings: dict[str, StringFit]  # fitted, per note
    manual_finger_width_mm: dict[str, float] = field(default_factory=dict)
    manual: dict[str, StringManual] = field(default_factory=dict)

    # --- Effective values --------------------------------------------------------

    def response_at(self, freqs: np.ndarray) -> np.ndarray:
        """H(f) in dB, interpolated in log frequency; held constant beyond the fitted range."""
        freqs = np.asarray(freqs, dtype=float)
        return np.interp(np.log(freqs), np.log(self.response_freqs), self.response_db)

    def finger_width(self, finger: str | None) -> float | None:
        if finger is None:
            return None
        return self.manual_finger_width_mm.get(finger, self.finger_width_mm.get(finger))

    def manual_for(self, note: str) -> StringManual:
        return self.manual.setdefault(note, StringManual())

    def damping(self, note: str) -> tuple[float, float]:
        fit, manual = self.strings[note], self.manual.get(note, StringManual())
        return (
            manual.sigma0 if manual.sigma0 is not None else fit.sigma0,
            manual.sigma2 if manual.sigma2 is not None else fit.sigma2,
        )

    def inharmonicity(self, note: str) -> float | None:
        manual = self.manual.get(note, StringManual())
        if manual.inharmonicity_b is not None:
            return manual.inharmonicity_b
        return self.strings[note].inharmonicity_b

    def decay_rates(self, note: str, harmonics: np.ndarray, freqs: np.ndarray) -> np.ndarray:
        """Per-harmonic decay rates: measured where available, else sigma0 + sigma2 f^2.

        If sigma0 or sigma2 has been adjusted by hand, the formula is used for
        every harmonic, so the adjustment takes effect everywhere.
        """
        sigma0, sigma2 = self.damping(note)
        rates = sigma0 + sigma2 * np.asarray(freqs, dtype=float) ** 2
        manual = self.manual.get(note, StringManual())
        if manual.sigma0 is None and manual.sigma2 is None:
            measured = self.strings[note].sigma_n
            for i, n in enumerate(harmonics):
                if n - 1 < len(measured) and measured[n - 1] is not None:
                    rates[i] = measured[n - 1]
        return rates

    def offsets_db(self, note: str, harmonics: np.ndarray) -> np.ndarray:
        offsets = self.manual.get(note, StringManual()).offsets_db
        return np.array([offsets.get(int(n), 0.0) for n in harmonics])

    def apply_to_string(self, string: StringParams) -> StringParams:
        """The string with this guitar's tuning, length, damping, and stiffness."""
        if string.note not in self.strings:
            return dataclasses.replace(string, length=self.scale_length_mm / 1000)
        sigma0, sigma2 = self.damping(string.note)
        return dataclasses.replace(
            string,
            f0=self.strings[string.note].f0,
            length=self.scale_length_mm / 1000,
            sigma0=sigma0,
            sigma2=sigma2,
            inharmonicity_b=self.inharmonicity(string.note),
        )

    # --- Refit and persistence ------------------------------------------------------

    def with_manual_from(self, previous: "GuitarProfile") -> "GuitarProfile":
        """This (newly fitted) profile, keeping the hand adjustments of ``previous``."""
        return dataclasses.replace(
            self,
            manual_finger_width_mm=dict(previous.manual_finger_width_mm),
            manual={
                k: dataclasses.replace(v, offsets_db=dict(v.offsets_db))
                for k, v in previous.manual.items()
                if not v.is_empty()
            },
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "scale_length_mm": self.scale_length_mm,
            "string_set": self.string_set,
            "recorder": self.recorder,
            "created": self.created,
            "source_measurements": self.source_measurements,
            "response_curve": {"freq_hz": self.response_freqs, "gain_db": self.response_db},
            "finger_width_mm": {
                "fitted": self.finger_width_mm,
                "manual": self.manual_finger_width_mm,
            },
            "strings": {
                note: {
                    "fitted": dataclasses.asdict(fit),
                    "manual": _manual_to_dict(self.manual.get(note, StringManual())),
                }
                for note, fit in self.strings.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GuitarProfile":
        strings, manual = {}, {}
        for note, entry in data["strings"].items():
            strings[note] = StringFit(**entry["fitted"])
            m = entry.get("manual") or {}
            manual[note] = StringManual(
                sigma0=m.get("sigma0"),
                sigma2=m.get("sigma2"),
                inharmonicity_b=m.get("inharmonicity_b"),
                offsets_db={int(k): float(v) for k, v in m.get("offsets_db", {}).items()},
            )
        return cls(
            name=data["name"],
            scale_length_mm=data["scale_length_mm"],
            string_set=data["string_set"],
            recorder=data.get("recorder"),
            created=data["created"],
            source_measurements=data.get("source_measurements", []),
            response_freqs=data["response_curve"]["freq_hz"],
            response_db=data["response_curve"]["gain_db"],
            finger_width_mm=data["finger_width_mm"]["fitted"],
            manual_finger_width_mm=data["finger_width_mm"].get("manual", {}),
            strings=strings,
            manual={k: v for k, v in manual.items() if not v.is_empty()},
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))


def _manual_to_dict(manual: StringManual) -> dict:
    out = {k: v for k, v in dataclasses.asdict(manual).items() if v not in (None, {})}
    if "offsets_db" in out:
        out["offsets_db"] = {str(k): v for k, v in sorted(manual.offsets_db.items())}
    return out


def profile_path(name_or_path: str | Path, directory: Path = PROFILE_DIR) -> Path:
    """A ``.json`` path as given, or ``<directory>/<name>.json`` for a bare name."""
    path = Path(name_or_path)
    return path if path.suffix == ".json" else directory / f"{name_or_path}.json"


def load_profile(
    name_or_path: str | Path, directory: Path = PROFILE_DIR, *, shared: bool = False
) -> GuitarProfile:
    """Load a profile by name (from ``directory``) or path.

    Loads are cached until the file changes. Each call returns an independent
    copy unless ``shared`` is set, for read-only use where speed matters.
    """
    path = profile_path(name_or_path, directory)
    if not path.is_file():
        raise FileNotFoundError(f"No guitar profile at {path}")
    profile = _load_cached(path.resolve(), path.stat().st_mtime)
    return profile if shared else copy.deepcopy(profile)


@lru_cache(maxsize=8)
def _load_cached(path: Path, mtime: float) -> GuitarProfile:
    return GuitarProfile.from_dict(json.loads(path.read_text()))
