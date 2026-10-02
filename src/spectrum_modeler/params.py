"""Parameter types shared by the model, synthesis, and analysis code.

Units: frequencies in Hz, string length in metres, tension in newtons, decay
rates in 1/s. Pluck distance and finger width are in millimetres from the
saddle, matching how they are measured on the guitar.
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .profile import GuitarProfile

Observable = Literal["displacement", "velocity", "bridge_force"]
OBSERVABLES: tuple[Observable, ...] = ("displacement", "velocity", "bridge_force")
# "full": sigma0 + sigma2 f^2; "flat": sigma0 only, so every partial decays
# together; "off": no decay at all.
Damping = Literal["full", "flat", "off"]
DAMPING_MODES: tuple[Damping, ...] = ("full", "flat", "off")


@dataclass(frozen=True)
class StringParams:
    note: str  # "E2" ... "E4"
    f0: float  # Hz
    length: float = 0.650  # m
    tension: float | None = None  # N, from the string set
    diameter: float | None = None  # m, optional
    youngs_modulus: float | None = None  # Pa, optional
    inharmonicity_b: float | None = None  # B, if known directly (e.g. from calibration)
    # Damping: mode at frequency f decays at sigma0 + sigma2 * f^2 (design §2.3a).
    # Defaults are rough nylon values: a 60 dB decay takes ~7 s for a low fundamental,
    # well under 1 s by 3 kHz.
    sigma0: float = 1.0  # 1/s
    sigma2: float = 2e-6  # 1/(s Hz^2)

    @property
    def linear_density(self) -> float | None:
        """Mass per unit length in kg/m, from mu = T / (2 L f0)^2."""
        if self.tension is None:
            return None
        return self.tension / (2 * self.length * self.f0) ** 2

    @property
    def b_coefficient(self) -> float | None:
        """Inharmonicity B: given directly, else pi^3 E d^4 / (64 T L^2), else None."""
        if self.inharmonicity_b is not None:
            return self.inharmonicity_b
        if None in (self.diameter, self.youngs_modulus, self.tension):
            return None
        return (
            math.pi**3
            * self.youngs_modulus
            * self.diameter**4
            / (64 * self.tension * self.length**2)
        )


@dataclass(frozen=True)
class PluckParams:
    distance_mm: float  # from saddle
    amplitude: float = 0.002  # m, pluck height
    width_mm: float = 0.0  # 0 = ideal point pluck
    finger: str | None = None  # "flesh", "nail", ...: a guitar profile may supply the width

    def beta(self, length: float) -> float:
        """Pluck position as a fraction of the string length (0 < beta < 1)."""
        beta = self.distance_mm / 1000 / length
        if not 0 < beta < 1:
            raise ValueError(
                f"Pluck distance {self.distance_mm} mm is outside the {length * 1000:g} mm string"
            )
        return beta


@dataclass(frozen=True)
class ModelOptions:
    n_harmonics: int = 40
    observable: Observable = "bridge_force"
    # Realism layers (design §2.3). None means automatic: on when a guitar
    # profile supplies the values, otherwise off.
    damping: Damping | bool = "full"  # True and False mean "full" and "off"
    finger_width: bool = True  # has no effect for an ideal (zero-width) pluck
    inharmonicity: bool | None = None  # needs the string's B
    body: bool | None = None  # calibrated response with a profile; else illustrative resonances
    # A guitar profile (name, path, or loaded profile) overrides f0, damping, B,
    # finger widths, and body response with calibrated values (design §4.7).
    profile: "str | Path | GuitarProfile | None" = None
    # Replace the string's (or profile's) sigma0 and sigma2. With a profile, an
    # override uses the formula for every harmonic instead of the measured rates.
    sigma0: float | None = None  # 1/s
    sigma2: float | None = None  # 1/(s Hz^2)

    def __post_init__(self) -> None:
        if self.n_harmonics < 1:
            raise ValueError("n_harmonics must be at least 1")
        if self.observable not in OBSERVABLES:
            raise ValueError(f"observable must be one of {OBSERVABLES}, not {self.observable!r}")
        if isinstance(self.damping, bool):
            object.__setattr__(self, "damping", "full" if self.damping else "off")
        if self.damping not in DAMPING_MODES:
            raise ValueError(f"damping must be one of {DAMPING_MODES}, not {self.damping!r}")
        for name in ("sigma0", "sigma2"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} can't be negative")
