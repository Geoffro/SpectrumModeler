"""Manual adjustment of a guitar profile against measured takes (design §4.6).

``TunerState`` holds the logic and no GUI, so it can be tested; ``viz/tuner.py``
puts a window on it. Adjustments are written to the profile's manual section
only, so the fitted values stay intact and a refit keeps the adjustments.
"""

from pathlib import Path

import numpy as np

from .audio import match_loudness
from .measure import Measurement, align_model_db, take_audio
from .modal import ModalSpectrum, coefficients
from .params import ModelOptions, PluckParams
from .presets import open_string
from .profile import GuitarProfile
from .recording import SAMPLE_RATE, Session
from .synthesis import render_spectrum

MIN_HARMONICS = 3


class TunerState:
    def __init__(
        self,
        profile: GuitarProfile,
        measurements: list[Measurement],
        profile_path: str | Path,
        session: Session | None = None,
    ) -> None:
        self.takes = [
            m
            for m in measurements
            if m.note in profile.strings and m.pluck_mm and m.n_reliable >= MIN_HARMONICS
        ]
        if not self.takes:
            raise ValueError("No measured takes match the strings in this profile")
        self.profile = profile
        self.path = Path(profile_path)
        self.session = session
        self.index = 0
        self.position_shift_mm = 0.0  # for viewing only: not saved
        self.dirty = False

    # --- Selection --------------------------------------------------------------------

    @property
    def current(self) -> Measurement:
        return self.takes[self.index]

    @property
    def note(self) -> str:
        return self.current.note

    def select(self, index: int) -> None:
        self.index = index % len(self.takes)
        self.position_shift_mm = 0.0

    def step(self, delta: int) -> None:
        self.select(self.index + delta)

    # --- Model against measurement -------------------------------------------------------

    def model(self, damping: bool = False) -> ModalSpectrum:
        m = self.current
        return coefficients(
            open_string(m.note, string_set=self.profile.string_set),
            PluckParams(distance_mm=m.pluck_mm + self.position_shift_mm, finger=m.finger),
            ModelOptions(n_harmonics=int(m.harmonics[-1]), damping=damping, profile=self.profile),
        )

    def levels(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(harmonics, measured dB, calibrated model dB aligned to it, measured mask).

        The mask marks harmonics with a measured onset level.
        """
        m = self.current
        measured = m.reliable & np.isfinite(m.initial_db)
        return m.harmonics, m.initial_db, align_model_db(m, self.model()), measured

    def rms_error_db(self) -> float:
        _, measured, model, has_level = self.levels()
        ok = has_level & np.isfinite(model)
        return float(np.sqrt(np.mean((measured[ok] - model[ok]) ** 2)))

    # --- Effective values (fitted unless adjusted) -----------------------------------------

    def damping(self) -> tuple[float, float]:
        return self.profile.damping(self.note)

    def inharmonicity(self) -> float | None:
        return self.profile.inharmonicity(self.note)

    def finger_width(self) -> float:
        return self.profile.finger_width(self.current.finger) or 0.0

    def offset(self, harmonic: int) -> float:
        manual = self.profile.manual.get(self.note)
        return manual.offsets_db.get(harmonic, 0.0) if manual else 0.0

    # --- Adjustments ------------------------------------------------------------------------

    def set_damping(self, sigma0: float | None = None, sigma2: float | None = None) -> None:
        manual = self.profile.manual_for(self.note)
        if sigma0 is not None:
            manual.sigma0 = sigma0
        if sigma2 is not None:
            manual.sigma2 = sigma2
        self.dirty = True

    def set_inharmonicity(self, b: float) -> None:
        self.profile.manual_for(self.note).inharmonicity_b = b
        self.dirty = True

    def set_finger_width(self, width_mm: float) -> None:
        self.profile.manual_finger_width_mm[self.current.finger] = width_mm
        self.dirty = True

    def set_offset(self, harmonic: int, db: float) -> None:
        offsets = self.profile.manual_for(self.note).offsets_db
        if db == 0:
            offsets.pop(harmonic, None)
        else:
            offsets[harmonic] = db
        self.dirty = True

    def match_harmonic(self, harmonic: int) -> float:
        """Set the offset that makes the model meet this take's measurement; returns it."""
        harmonics, measured, model, has_level = self.levels()
        i = int(np.flatnonzero(harmonics == harmonic)[0])
        if not has_level[i] or not np.isfinite(model[i]):
            raise ValueError(f"Harmonic {harmonic} was not measured in this take")
        new = round(self.offset(harmonic) + measured[i] - model[i], 1)
        self.set_offset(harmonic, new)
        return new

    def clear_string(self) -> None:
        """Drop every manual adjustment for the current string (and its finger width)."""
        self.profile.manual.pop(self.note, None)
        self.profile.manual_finger_width_mm.pop(self.current.finger, None)
        self.dirty = True

    def save(self) -> Path:
        self.profile.save(self.path)
        self.dirty = False
        return self.path

    # --- Audio ------------------------------------------------------------------------------

    def ab_audio(self) -> tuple[np.ndarray, np.ndarray, int]:
        """(recording, calibrated model) for this take, matched in loudness."""
        if self.session is None:
            raise ValueError("No session, so no recording to play")
        recording = take_audio(self.session, self.current)
        duration = recording.size / SAMPLE_RATE
        model = render_spectrum(self.model(damping=True), duration, SAMPLE_RATE, normalize=False)
        recording, model = match_loudness(recording, model)
        return recording, model, SAMPLE_RATE
