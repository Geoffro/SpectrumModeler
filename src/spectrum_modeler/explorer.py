"""Explorer logic: one pluck's model under adjustable settings, with A/B comparison.

``ExplorerState`` has no GUI, so it can be tested; ``viz/explorer.py`` puts a
window on it. "Pinning" freezes the current settings as B, so any two
configurations (positions, fingers, strings, calibrated or not) can be
compared by eye and by ear.
"""

import dataclasses
from dataclasses import dataclass

import numpy as np

from .analysis import spectral_centroid
from .audio import match_loudness, sequence
from .measure import Measurement, take_audio
from .modal import ModalSpectrum, coefficients, missing_harmonics
from .params import DAMPING_MODES, Damping, ModelOptions, StringParams
from .presets import DEFAULT_SCALE_LENGTH, OPEN_STRING_NOTES, open_strings, pluck
from .profile import GuitarProfile
from .recording import SAMPLE_RATE, Session
from .string_sets import DEFAULT_STRING_SET
from .synthesis import render_spectrum

DISPLAY_HARMONICS = 40
AUDIO_MAX_FREQ = 12_000.0  # Hz, highest partial rendered for listening
EDGE_MM = 10.0  # closest a pluck may be to the saddle or nut
RING_HARMONICS = (1, 10, DISPLAY_HARMONICS)  # ring times shown in the readout


@dataclass(frozen=True)
class Settings:
    note: str = "E2"
    position_mm: float = 130.0
    finger: str = "flesh"
    damping: Damping = "full"
    finger_width: bool = True
    inharmonicity: bool = False
    body: bool = False
    calibrated: bool = False
    # None: the string's own value (the profile's, when calibrated)
    sigma0: float | None = None  # 1/s
    sigma2: float | None = None  # 1/(s Hz^2)

    def __post_init__(self) -> None:
        if self.damping not in DAMPING_MODES:
            raise ValueError(f"damping must be one of {DAMPING_MODES}, not {self.damping!r}")
        for name in ("sigma0", "sigma2"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} can't be negative")

    def describe(self) -> str:
        mode = "calibrated" if self.calibrated else "uncalibrated"
        text = f"{self.note}, {self.position_mm:g} mm, {self.finger}, {mode}"
        if self.damping != "full":
            text += ", no damping" if self.damping == "off" else ", flat damping"
        if self.damping != "off":
            if self.sigma0 is not None:
                text += f", σ0 {self.sigma0:.2f}"
            if self.sigma2 is not None and self.damping == "full":
                text += f", σ2 {self.sigma2:.1e}"
        return text


class ExplorerState:
    def __init__(
        self,
        *,
        string_set: str = DEFAULT_STRING_SET,
        profile: GuitarProfile | None = None,
        session: Session | None = None,
        measurements: list[Measurement] | None = None,
    ) -> None:
        self.profile = profile
        self.length = profile.scale_length_mm / 1000 if profile else DEFAULT_SCALE_LENGTH
        self.strings: dict[str, StringParams] = {
            s.note: s for s in open_strings(string_set=string_set, length=self.length)
        }
        self.session = session
        self.recordings = [m for m in measurements or [] if m.note and m.pluck_mm]
        self.settings = Settings(calibrated=profile is not None)
        if profile is not None:
            self.settings = dataclasses.replace(self.settings, inharmonicity=True, body=True)
        self.pinned: Settings | None = None

    # --- Changing settings ---------------------------------------------------------------

    @property
    def max_position_mm(self) -> float:
        return self.length * 1000 - EDGE_MM

    def update(self, **changes) -> None:
        if "note" in changes and changes["note"] not in OPEN_STRING_NOTES:
            raise ValueError(f"Unknown open string {changes['note']!r}")
        if "position_mm" in changes:
            changes["position_mm"] = float(
                np.clip(changes["position_mm"], EDGE_MM, self.max_position_mm)
            )
        if changes.get("calibrated"):
            if self.profile is None:
                raise ValueError("No guitar profile loaded")
            # a profile's stiffness and body response are the point of calibrating
            changes.setdefault("inharmonicity", True)
            changes.setdefault("body", True)
        self.settings = dataclasses.replace(self.settings, **changes)

    def pin(self) -> None:
        self.pinned = self.settings

    def unpin(self) -> None:
        self.pinned = None

    # --- Model ---------------------------------------------------------------------------------

    def _options(self, s: Settings, n_harmonics: int) -> tuple[ModelOptions, bool]:
        """Model options for these settings, and whether stiffness could be applied."""
        profile = self.profile if s.calibrated else None
        has_b = (
            profile is not None
            and s.note in profile.strings
            and profile.inharmonicity(s.note) is not None
        ) or self.strings[s.note].b_coefficient is not None
        opts = ModelOptions(
            n_harmonics=n_harmonics,
            damping=s.damping,
            finger_width=s.finger_width,
            inharmonicity=s.inharmonicity and has_b,
            body=s.body,
            profile=profile,
            sigma0=s.sigma0,
            sigma2=s.sigma2,
        )
        return opts, has_b

    def spectrum(
        self, settings: Settings | None = None, n_harmonics: int = DISPLAY_HARMONICS
    ) -> ModalSpectrum:
        s = settings or self.settings
        opts, _ = self._options(s, n_harmonics)
        return coefficients(
            self.strings[s.note], pluck(s.position_mm, length=self.length, finger=s.finger), opts
        )

    def damping_values(self, settings: Settings | None = None) -> tuple[float, float]:
        """The sigma0 and sigma2 in use: the overrides, else the string's (or profile's)."""
        string = self.spectrum(settings, 1).string
        return string.sigma0, string.sigma2

    def ring_times(self, settings: Settings | None = None) -> dict[int, float]:
        """Seconds for harmonics in ``RING_HARMONICS`` to fall 60 dB (inf: no decay)."""
        spectrum = self.spectrum(settings)
        with np.errstate(divide="ignore"):
            return {n: float(np.log(1000) / spectrum.decay_rates[n - 1]) for n in RING_HARMONICS}

    def pinned_spectrum(self) -> ModalSpectrum | None:
        return self.spectrum(self.pinned) if self.pinned else None

    def centroid(self, settings: Settings | None = None) -> float:
        spectrum = self.spectrum(settings)
        return spectral_centroid(spectrum.freqs, spectrum.amps)

    def notices(self) -> list[str]:
        """Why a requested layer is not in effect, if it isn't."""
        s = self.settings
        _, has_b = self._options(s, 1)
        notes = []
        if s.inharmonicity and not has_b:
            notes.append(f"no stiffness (B) for {s.note}: add it to the string set, or calibrate")
        return notes

    def readout(self) -> list[str]:
        s = self.settings
        beta = s.position_mm / 1000 / self.length
        missing = missing_harmonics(beta, DISPLAY_HARMONICS)
        lines = [
            f"β = {beta:.3f}",
            f"brightness (spectral centroid): {self.centroid():.0f} Hz",
            "missing harmonics: "
            + (", ".join(map(str, missing)) if missing.size else f"none ≤ {DISPLAY_HARMONICS}"),
        ]
        if s.damping == "off":
            lines.append("ring time (−60 dB): forever (damping off)")
        else:
            times = " · ".join(
                f"h{n} {t:.1f} s" if np.isfinite(t) else f"h{n} forever"
                for n, t in self.ring_times().items()
            )
            measured = (
                s.damping == "full" and s.calibrated and s.sigma0 is None and s.sigma2 is None
                and self.profile.uses_measured_decay(s.note)
            )  # fmt: skip
            lines.append(f"ring time (−60 dB): {times}" + (" (measured)" if measured else ""))
        if self.pinned:
            lines.append(
                f"B: {self.pinned.describe()}, centroid {self.centroid(self.pinned):.0f} Hz"
            )
        nearest = self.nearest_recording()
        if nearest is not None:
            lines.append(f"nearest recording: {nearest.clip} t{nearest.take}")
        return lines + self.notices()

    # --- Audio ---------------------------------------------------------------------------------

    def audio(self, settings: Settings | None = None, duration: float = 3.0) -> np.ndarray:
        """The model pluck, unnormalized (bridge-force units)."""
        s = settings or self.settings
        f0 = self.strings[s.note].f0
        n = min(100, int(AUDIO_MAX_FREQ // f0))
        return render_spectrum(self.spectrum(s, n), duration, SAMPLE_RATE, normalize=False)

    def playable(self, duration: float = 3.0) -> np.ndarray:
        """The current model pluck at a comfortable level."""
        (audio,) = match_loudness(self.audio(duration=duration))
        return audio

    def ab_audio(self, duration: float = 3.0, gap: float = 0.5) -> np.ndarray:
        """A (current) then B (pinned), matched in loudness."""
        if self.pinned is None:
            raise ValueError("Pin a setting as B first")
        a, b = match_loudness(self.audio(duration=duration), self.audio(self.pinned, duration))
        return sequence([a, b], SAMPLE_RATE, gap)

    def nearest_recording(self) -> Measurement | None:
        """The recorded take of this string closest to the current pluck position."""
        s = self.settings
        same = [m for m in self.recordings if m.note == s.note]
        if not same:
            return None
        return min(same, key=lambda m: (abs(m.pluck_mm - s.position_mm), m.take or 0))

    def recording_audio(self) -> np.ndarray:
        nearest = self.nearest_recording()
        if nearest is None or self.session is None:
            raise ValueError(f"No recording of {self.settings.note} loaded")
        (audio,) = match_loudness(take_audio(self.session, nearest))
        return audio
