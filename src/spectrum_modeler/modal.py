"""Modal solution for a plucked string (design §2.1-2.3).

The string is fixed at x = 0 (saddle) and x = L (nut), pulled to height h at
x_p = beta * L, and released from rest. Mode n has shape sin(n pi x / L),
frequency n f0, and displacement amplitude

    a_n = 2 h sin(n pi beta) / (n^2 pi^2 beta (1 - beta)).

Amplitudes are signed; take the magnitude for plotting.
"""

import dataclasses
from dataclasses import dataclass

import numpy as np

from . import body
from .params import ModelOptions, Observable, PluckParams, StringParams
from .profile import GuitarProfile, load_profile

UNITS: dict[Observable, str] = {"displacement": "m", "velocity": "m/s", "bridge_force": "N"}

# How close n * beta must be to a whole number for mode n to count as missing.
# Tight on purpose: a pluck 0.01 mm off a node still excites the mode slightly.
NODE_TOLERANCE = 1e-9


@dataclass(frozen=True, eq=False)
class ModalSpectrum:
    string: StringParams
    pluck: PluckParams
    observable: Observable
    harmonics: np.ndarray  # n = 1, 2, ...
    freqs: np.ndarray  # Hz
    amps: np.ndarray  # signed, in UNITS[observable]
    decay_rates: np.ndarray  # 1/s; zero for the lossless ideal string

    @property
    def beta(self) -> float:
        return self.pluck.beta(self.string.length)

    @property
    def units(self) -> str:
        return UNITS[self.observable]

    def levels_db(self, ref: float | None = None, floor_db: float = -120.0) -> np.ndarray:
        """Magnitudes in dB relative to ``ref`` (default: the strongest harmonic)."""
        mags = np.abs(self.amps)
        ref = mags.max() if ref is None else ref
        with np.errstate(divide="ignore"):
            levels = 20 * np.log10(mags / ref)
        return np.maximum(levels, floor_db)


def _at_node(n: np.ndarray, beta: float) -> np.ndarray:
    nb = n * beta
    return np.abs(nb - np.round(nb)) < NODE_TOLERANCE


def displacement_coefficients(n: np.ndarray, beta: float, amplitude: float) -> np.ndarray:
    """a_n for harmonic numbers ``n``; exactly zero where the pluck is on a node."""
    if not 0 < beta < 1:
        raise ValueError(f"beta must be between 0 and 1, not {beta}")
    n = np.asarray(n, dtype=float)
    b = 2 * amplitude * np.sin(n * np.pi * beta) / (n**2 * np.pi**2 * beta * (1 - beta))
    b[_at_node(n, beta)] = 0.0
    return b


def finger_width_factor(n: np.ndarray, width: float, length: float) -> np.ndarray:
    """S_n = sinc(n pi w / 2L): a finger ``width`` m wide rounding off the corner."""
    return np.sinc(np.asarray(n, dtype=float) * width / (2 * length))


def _resolve_profile(profile) -> GuitarProfile | None:
    if profile is None or isinstance(profile, GuitarProfile):
        return profile
    return load_profile(profile, shared=True)


def coefficients(
    string: StringParams, pluck: PluckParams, opts: ModelOptions | None = None
) -> ModalSpectrum:
    """Mode frequencies, amplitudes, and decay rates for the chosen observable.

    - ``displacement``: a_n (m)
    - ``velocity``: modal velocity amplitude 2 pi f_n a_n (m/s)
    - ``bridge_force``: F_n = T a_n n pi / L (N); needs ``string.tension``

    Realism layers switched on in ``opts`` are applied on top (design §2.3). With
    a guitar profile, its calibrated values replace the defaults: tuning, length,
    damping (measured per harmonic where available), stiffness, finger width,
    body response, and any per-harmonic manual offsets. ``opts.sigma0`` and
    ``opts.sigma2`` override the string's (or profile's); the returned ``string``
    carries the values used.
    """
    opts = opts or ModelOptions()
    profile = _resolve_profile(opts.profile)
    if profile is not None:
        string = profile.apply_to_string(string)
        width = profile.finger_width(pluck.finger)
        if width is not None:
            pluck = dataclasses.replace(pluck, width_mm=width)
    calibrated = profile is not None and string.note in profile.strings
    adjusted = opts.sigma0 is not None or opts.sigma2 is not None
    if adjusted:
        string = dataclasses.replace(
            string,
            sigma0=string.sigma0 if opts.sigma0 is None else opts.sigma0,
            sigma2=string.sigma2 if opts.sigma2 is None else opts.sigma2,
        )

    beta = pluck.beta(string.length)
    n = np.arange(1, opts.n_harmonics + 1)

    freqs = n * string.f0
    use_b = opts.inharmonicity
    if use_b is None:
        use_b = calibrated and string.b_coefficient is not None
    if use_b:
        b_coef = string.b_coefficient
        if b_coef is None:
            raise ValueError(
                f"The inharmonicity layer needs B for the {string.note} string: set "
                "inharmonicity_b, or diameter_mm and youngs_modulus_gpa, in the string set"
            )
        freqs = freqs * np.sqrt(1 + b_coef * n**2)

    b = displacement_coefficients(n, beta, pluck.amplitude)
    if opts.finger_width and pluck.width_mm > 0:
        b = b * finger_width_factor(n, pluck.width_mm / 1000, string.length)

    match opts.observable:
        case "displacement":
            amps = b
        case "velocity":
            amps = 2 * np.pi * freqs * b
        case "bridge_force":
            if string.tension is None:
                raise ValueError(f"bridge_force needs the {string.note} string's tension")
            amps = string.tension * b * n * np.pi / string.length

    use_body = opts.body if opts.body is not None else profile is not None
    if use_body:
        gain_db = profile.response_at(freqs) if profile is not None else body.response_db(freqs)
        amps = amps * 10 ** (gain_db / 20)
    if calibrated:
        amps = amps * 10 ** (profile.offsets_db(string.note, n) / 20)

    match opts.damping:
        case "off":
            decay_rates = np.zeros_like(freqs, dtype=float)
        case "flat":
            decay_rates = np.full_like(freqs, string.sigma0, dtype=float)
        case "full" if calibrated and not adjusted:
            decay_rates = profile.decay_rates(string.note, n, freqs)
        case "full":
            decay_rates = string.sigma0 + string.sigma2 * freqs**2

    return ModalSpectrum(
        string=string,
        pluck=pluck,
        observable=opts.observable,
        harmonics=n,
        freqs=freqs,
        amps=amps,
        decay_rates=decay_rates,
    )


def missing_harmonics(beta: float, n_max: int) -> np.ndarray:
    """Harmonic numbers up to ``n_max`` with a node at the pluck point (n * beta whole)."""
    n = np.arange(1, n_max + 1)
    return n[_at_node(n, beta)]


def envelope_nulls(beta: float, n_max: float) -> np.ndarray:
    """Where |sin(n pi beta)| is zero: n = k / beta, up to ``n_max``.

    These need not be whole numbers; harmonics near them are weak even when
    not missing outright.
    """
    k = np.arange(1, np.floor(n_max * beta + NODE_TOLERANCE) + 1)
    return k / beta


def initial_shape(x: np.ndarray, length: float, pluck: PluckParams) -> np.ndarray:
    """The triangular string shape at release, for positions ``x`` in m from the saddle."""
    x = np.asarray(x, dtype=float)
    xp = pluck.beta(length) * length
    h = pluck.amplitude
    return np.where(x <= xp, h * x / xp, h * (length - x) / (length - xp))


def mode_contributions(
    x: np.ndarray, string: StringParams, pluck: PluckParams, n_modes: int
) -> np.ndarray:
    """Each mode's displacement a_n sin(n pi x / L) at release; shape (n_modes, len(x))."""
    n = np.arange(1, n_modes + 1)
    b = displacement_coefficients(n, pluck.beta(string.length), pluck.amplitude)
    x = np.asarray(x, dtype=float)
    return b[:, None] * np.sin(np.outer(n, x) * np.pi / string.length)
