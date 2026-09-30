"""Calibration: fit the model to measured plucks and produce a guitar profile (design §4.5).

The measured onset level of harmonic n in take t is modeled in dB as

    M[t, n] = G[t] + T_n(beta[t], w[finger]) + H(f[t, n])

- T_n: the bridge-force pluck spectrum, 20 log10 |sin(n pi beta) / n * sinc(n pi w / 2L)|
- G[t]: the level of each take (pluck force and gain are uncontrolled)
- H(f): one response curve shared by every string, a cubic B-spline in
  log frequency with a roughness penalty
- w: one effective finger width per finger type
- beta[t]: each pluck's position, refined within +/- MAX_BETA_SHIFT_MM of the manifest

H depends on frequency but not on pluck position, so recording each string at
several positions separates it from T_n: the node pattern moves with position,
body resonances stay put. Everything is fitted jointly by robust least squares.
Damping, stiffness, and tuning come straight from the measurements, per string.
"""

import datetime
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.interpolate import BSpline
from scipy.optimize import least_squares

from .measure import Measurement, fit_damping
from .presets import DEFAULT_SCALE_LENGTH, FINGER_WIDTHS_MM
from .profile import GuitarProfile, StringFit
from .string_sets import DEFAULT_STRING_SET

MAX_BETA_SHIFT_MM = 5.0
POSITION_STEP_MM = 0.1  # grid for refining each take's pluck position
POSITION_PRIOR_MM = 2.0  # how far a tape mark is typically off
NODE_DEPTH_DB = 40.0  # deeper than real plucks usually land, shallow enough to stay robust
KNOTS_PER_OCTAVE = 4
SMOOTHING = 1.0  # weight of the roughness penalty on H
ROBUST_SCALE_DB = 3.0  # residuals beyond this count less (soft L1)
MIN_TAKE_HARMONICS = 3
RESPONSE_POINTS = 200
MAX_EVALUATIONS = 200  # residual evaluations per solve; a guard, not normally reached
POSITION_ROUNDS = 2


@dataclass(frozen=True)
class TakeFit:
    clip: str
    take: int | None
    note: str
    finger: str
    manifest_mm: float
    fitted_mm: float
    gain_db: float
    rms_db: float  # of this take's residuals
    n_harmonics: int


@dataclass(frozen=True, eq=False)
class CalibrationResult:
    profile: GuitarProfile
    takes: list[TakeFit]
    # One entry per data point (a reliable harmonic of a take):
    freqs: np.ndarray  # Hz
    notes: np.ndarray  # note of each point
    measured_db: np.ndarray
    predicted_db: np.ndarray
    response_data_db: np.ndarray  # measured minus G and T: H as seen by each point

    @property
    def rms_db(self) -> float:
        return float(np.sqrt(np.mean((self.measured_db - self.predicted_db) ** 2)))


def _knot_vector(knots_lo: float, knots_hi: float, per_octave: float) -> np.ndarray:
    """Clamped cubic B-spline knot vector over log2 frequency."""
    n_interior = max(1, int(np.ceil((knots_hi - knots_lo) * per_octave)) - 1)
    interior = np.linspace(knots_lo, knots_hi, n_interior + 2)[1:-1]
    return np.concatenate([[knots_lo] * 4, interior, [knots_hi] * 4])


def pluck_spectrum_db(
    n: np.ndarray, beta: np.ndarray, width_mm: np.ndarray, length: float
) -> np.ndarray:
    """T_n in dB, with nodes at most NODE_DEPTH_DB below the 1/n envelope.

    Real plucks never land exactly on a node, so measured node harmonics are
    weak but finite; an unlimited model node would produce residuals of 70 dB
    that swamp the fit.
    """
    envelope = np.sinc(n * width_mm / 1000 / (2 * length)) / n
    shape = np.sin(n * np.pi * beta) * envelope
    floor = 10 ** (-NODE_DEPTH_DB / 20) * envelope
    return 10 * np.log10(shape**2 + floor**2)


def calibrate(
    measurements: list[Measurement],
    *,
    name: str,
    scale_length: float = DEFAULT_SCALE_LENGTH,
    string_set: str = DEFAULT_STRING_SET,
    recorder: str | None = None,
    refine_beta: bool = True,
    knots_per_octave: float = KNOTS_PER_OCTAVE,
    smoothing: float = SMOOTHING,
) -> CalibrationResult:
    """Fit a guitar profile to measurements of open-string plucks at known positions."""
    usable = [
        m
        for m in measurements
        if m.note and m.pluck_mm and m.finger and m.n_reliable >= MIN_TAKE_HARMONICS
    ]
    if not usable:
        raise ValueError("No measurements with a note, pluck position, and finger to calibrate")

    strings = _fit_strings(usable)

    # --- Data points: every reliable harmonic with an onset level -------------------
    take_idx, n, freqs, measured = [], [], [], []
    for t, m in enumerate(usable):
        ok = m.reliable & np.isfinite(m.initial_db)
        take_idx.append(np.full(ok.sum(), t))
        n.append(m.harmonics[ok])
        freqs.append(m.freqs[ok])
        measured.append(m.initial_db[ok])
    take_idx = np.concatenate(take_idx)
    n = np.concatenate(n).astype(float)
    freqs = np.concatenate(freqs)
    measured = np.concatenate(measured)

    fingers = sorted({m.finger for m in usable})
    finger_of_take = np.array([fingers.index(m.finger) for m in usable])
    manifest_beta = np.array([m.pluck_mm / 1000 / scale_length for m in usable])
    max_shift = MAX_BETA_SHIFT_MM / 1000 / scale_length

    # H spans the 1st-99th percentile of measured frequencies: a spline end
    # resting on a handful of points swings wildly. Points beyond still count,
    # evaluated at the ends; the profile holds H constant outside the range.
    log_lo, log_hi = np.percentile(np.log2(freqs), [1, 99])
    log_f = np.clip(np.log2(freqs), log_lo, log_hi)
    knots = _knot_vector(log_lo, log_hi, knots_per_octave)
    basis = BSpline.design_matrix(log_f, knots, 3).tocsr()
    n_coef = basis.shape[1]
    second_diff = np.diff(np.eye(n_coef), 2, axis=0)

    n_takes, n_fingers = len(usable), len(fingers)
    n_shift = n_takes if refine_beta else 0
    sl_gain = slice(0, n_takes)
    sl_coef = slice(n_takes, n_takes + n_coef)
    sl_width = slice(sl_coef.stop, sl_coef.stop + n_fingers)
    sl_shift = slice(sl_width.stop, sl_width.stop + n_shift)

    def unpack(p):
        beta = manifest_beta + (p[sl_shift] if refine_beta else 0.0)
        return p[sl_gain], p[sl_coef], p[sl_width], beta

    def predict(p):
        gain, coef, width, beta = unpack(p)
        pluck = pluck_spectrum_db(n, beta[take_idx], width[finger_of_take[take_idx]], scale_length)
        return gain[take_idx] + pluck + basis @ coef

    def residuals(p):
        coef = p[sl_coef]
        return np.concatenate([
            predict(p) - measured,
            np.sqrt(smoothing) * (second_diff @ coef),
            [10.0 * coef.mean()],  # H is defined up to a constant, which G absorbs; pin it
        ])  # fmt: skip

    # Which parameters each residual depends on, for fast finite differences.
    n_rows = measured.size + second_diff.shape[0] + 1
    pattern = sparse.lil_matrix((n_rows, sl_shift.stop), dtype=int)
    rows = np.arange(measured.size)
    pattern[rows, sl_gain.start + take_idx] = 1
    pattern[: measured.size, sl_coef] = basis.toarray() != 0
    pattern[rows, sl_width.start + finger_of_take[take_idx]] = 1
    if refine_beta:
        pattern[rows, sl_shift.start + take_idx] = 1
    pattern[measured.size :, sl_coef] = 1

    # Starting point: presets for widths, manifest positions, flat H, gains to match.
    width0 = np.array([FINGER_WIDTHS_MM.get(f, 5.0) for f in fingers])
    pluck0 = pluck_spectrum_db(
        n, manifest_beta[take_idx], width0[finger_of_take[take_idx]], scale_length
    )
    gain0 = np.array([np.median((measured - pluck0)[take_idx == t]) for t in range(n_takes)])
    p0 = np.concatenate([gain0, np.zeros(n_coef), width0, np.zeros(n_shift)])
    lower = np.concatenate([
        np.full(n_takes + n_coef, -np.inf), np.zeros(n_fingers), np.full(n_shift, -max_shift)
    ])  # fmt: skip
    upper = np.concatenate([
        np.full(n_takes + n_coef, np.inf), np.full(n_fingers, 40.0), np.full(n_shift, max_shift)
    ])  # fmt: skip

    pattern = pattern.tocsr()

    def solve(start: np.ndarray, free: np.ndarray) -> np.ndarray:
        """Least squares over the parameters in ``free``, the rest held at ``start``."""

        def partial(x):
            p = start.copy()
            p[free] = x
            return residuals(p)

        fit = least_squares(
            partial,
            start[free],
            jac_sparsity=pattern[:, free],
            bounds=(lower[free], upper[free]),
            loss="soft_l1",
            f_scale=ROBUST_SCALE_DB,
            x_scale="jac",
            max_nfev=MAX_EVALUATIONS,
        )
        p = start.copy()
        p[free] = fit.x
        return p

    def search_positions(p: np.ndarray) -> np.ndarray:
        """Best shift for each take on a grid, holding H and widths: a global search,
        since a pluck near a node has several near-equal local minima."""
        _, coef, width, _ = unpack(p)
        response = basis @ coef
        grid = np.arange(-max_shift, max_shift + 1e-12, POSITION_STEP_MM / 1000 / scale_length)
        shifts = np.zeros(n_takes)
        for t in range(n_takes):
            rows = take_idx == t
            pluck = pluck_spectrum_db(
                n[rows][None, :],
                manifest_beta[t] + grid[:, None],
                width[finger_of_take[t]],
                scale_length,
            )
            misfit = measured[rows] - response[rows] - pluck
            misfit -= np.median(misfit, axis=1, keepdims=True)  # best gain for each shift
            z = misfit / ROBUST_SCALE_DB
            cost = np.sum(np.sqrt(1 + z**2) - 1, axis=1)  # soft L1, as in the fit
            # The tape mark is evidence too: near a node several positions fit
            # almost equally well, and this keeps the choice near the mark.
            shift_mm = grid * scale_length * 1000
            cost += 0.5 * (shift_mm / POSITION_PRIOR_MM) ** 2
            shifts[t] = grid[np.argmin(cost)]
        return shifts

    # Fit H, gains, and widths at the manifest positions; then alternate a
    # position search per take with a refit of the rest.
    fixed_positions = np.arange(sl_width.stop)
    p = solve(p0, fixed_positions)
    if refine_beta:
        for _ in range(POSITION_ROUNDS):
            p[sl_shift] = search_positions(p)
            p = solve(p, fixed_positions)
    gain, coef, width, beta = unpack(p)
    predicted = predict(p)
    pluck_db = pluck_spectrum_db(n, beta[take_idx], width[finger_of_take[take_idx]], scale_length)

    curve_freqs = np.geomspace(2**log_lo, 2**log_hi, RESPONSE_POINTS)
    curve_db = BSpline(knots, coef, 3)(np.log2(curve_freqs))

    profile = GuitarProfile(
        name=name,
        scale_length_mm=scale_length * 1000,
        string_set=string_set,
        recorder=recorder,
        created=datetime.date.today().isoformat(),
        source_measurements=sorted({f"{m.clip}_t{m.take}" for m in usable}),
        response_freqs=curve_freqs.tolist(),
        response_db=curve_db.tolist(),
        finger_width_mm={f: float(w) for f, w in zip(fingers, width, strict=True)},
        strings=strings,
    )
    takes = [
        TakeFit(
            clip=m.clip,
            take=m.take,
            note=m.note,
            finger=m.finger,
            manifest_mm=m.pluck_mm,
            fitted_mm=float(beta[t] * scale_length * 1000),
            gain_db=float(gain[t]),
            rms_db=float(np.sqrt(np.mean((predicted - measured)[take_idx == t] ** 2))),
            n_harmonics=int((take_idx == t).sum()),
        )
        for t, m in enumerate(usable)
    ]
    notes = np.array([usable[t].note for t in take_idx])
    return CalibrationResult(
        profile=profile,
        takes=takes,
        freqs=freqs,
        notes=notes,
        measured_db=measured,
        predicted_db=predicted,
        response_data_db=measured - gain[take_idx] - pluck_db,
    )


def _fit_strings(measurements: list[Measurement]) -> dict[str, StringFit]:
    """Tuning, stiffness, and damping per string, pooled over all its takes."""
    by_note: dict[str, list[Measurement]] = defaultdict(list)
    for m in measurements:
        by_note[m.note].append(m)

    strings = {}
    for note, ms in by_note.items():
        freqs = np.concatenate([m.freqs for m in ms])
        rates = np.concatenate([m.decay_rates for m in ms])
        damping = fit_damping(freqs, rates)
        if damping is None:
            raise ValueError(f"Too few decay rates measured on {note} to fit damping")
        b_values = [m.inharmonicity_b for m in ms if m.inharmonicity_b is not None]

        n_max = max(m.harmonics[-1] for m in ms)
        per_harmonic = np.full((len(ms), n_max), np.nan)
        for i, m in enumerate(ms):
            per_harmonic[i, m.harmonics - 1] = m.decay_rates
        with np.errstate(all="ignore"):
            sigma_n = [
                None if np.all(np.isnan(col)) else float(np.nanmedian(col))
                for col in per_harmonic.T
            ]

        strings[note] = StringFit(
            f0=float(np.median([m.f0 for m in ms])),
            inharmonicity_b=float(np.median(b_values)) if b_values else None,
            sigma0=damping[0],
            sigma2=damping[1],
            sigma_n=sigma_n,
        )
    return strings
