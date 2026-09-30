"""Finite-difference time-domain simulation of the plucked string (design §2.4).

Solves, on a grid of N intervals of width h with time step k,

    y_tt = c^2 y_xx - kappa^2 y_xxxx - 2 sigma0 y_t + 2 sigma1 y_txx

with the ends fixed (and simply supported, for the stiffness term). The scheme
is the standard explicit one: centred differences in time and space, with the
frequency-dependent loss taken as a backward difference in time.

- The ideal string runs at the Courant limit, lambda = c k / h = 1, where the
  scheme is exact: every mode oscillates at exactly n f0.
- Stiffness (kappa) and frequency-dependent loss (sigma1) tighten the stability
  condition, so h is chosen just coarse enough to satisfy it.

The coefficients follow the modal model: sigma1 = sigma2 c^2 / (4 pi^2) gives a
decay rate of sigma0 + sigma2 f^2, and kappa^2 = B c^2 L^2 / pi^2 gives partials
at n f0 sqrt(1 + B n^2).
"""

from dataclasses import dataclass

import numpy as np

from .modal import finger_width_factor, initial_shape
from .params import PluckParams, StringParams


@dataclass(frozen=True, eq=False)
class FdtdResult:
    string: StringParams
    pluck: PluckParams
    x: np.ndarray  # m, grid points including both ends
    dt: float  # s
    bridge_force: np.ndarray  # N, one value per time step
    snapshot_times: np.ndarray  # s
    snapshots: np.ndarray  # (n_snapshots, n_points) displacement, m
    energy: np.ndarray | None  # J per step, if tracked
    courant: float  # lambda = c dt / h

    @property
    def sample_rate(self) -> float:
        return 1 / self.dt

    @property
    def times(self) -> np.ndarray:
        return np.arange(self.bridge_force.size) * self.dt


def _coefficients(
    string: StringParams, damping: bool, stiffness: bool
) -> tuple[float, float, float, float]:
    """(c, sigma0, sigma1, kappa) for the PDE."""
    c = 2 * string.length * string.f0
    sigma0 = string.sigma0 if damping else 0.0
    sigma1 = string.sigma2 * c**2 / (4 * np.pi**2) if damping else 0.0
    kappa = 0.0
    if stiffness:
        b = string.b_coefficient
        if b is None:
            raise ValueError(f"Stiffness needs B for the {string.note} string")
        kappa = np.sqrt(b) * c * string.length / np.pi
    return c, sigma0, sigma1, kappa


def min_spacing(c: float, k: float, sigma1: float, kappa: float) -> float:
    """Smallest stable grid spacing h for time step k (the Courant condition, generalized)."""
    a = c**2 * k**2 + 4 * sigma1 * k
    return float(np.sqrt((a + np.sqrt(a**2 + 16 * kappa**2 * k**2)) / 2))


def _initial_displacement(x: np.ndarray, string: StringParams, pluck: PluckParams) -> np.ndarray:
    """The plucked shape on the grid; a finger of nonzero width rounds the corner.

    The width is applied through the string's sine modes, exactly as in the modal
    model, so both start from the same shape.
    """
    shape = initial_shape(x, string.length, pluck)
    if pluck.width_mm <= 0:
        return shape
    n_modes = x.size - 1
    n = np.arange(1, n_modes)
    modes = np.sin(np.outer(n, x) * np.pi / string.length)
    coefficients = (2 / n_modes) * modes[:, 1:-1] @ shape[1:-1]  # discrete sine transform
    coefficients *= finger_width_factor(n, pluck.width_mm / 1000, string.length)
    return coefficients @ modes


def simulate(
    string: StringParams,
    pluck: PluckParams,
    duration: float,
    *,
    sample_rate: float = 48_000,
    damping: bool = False,
    stiffness: bool = False,
    n_intervals: int | None = None,
    snapshot_every: int | None = None,
    snapshot_until: float | None = None,
    track_energy: bool = False,
) -> FdtdResult:
    """Simulate ``duration`` seconds after release.

    ``n_intervals`` fixes the grid (an error if unstable); by default the finest
    stable grid is used. Every ``snapshot_every`` steps up to ``snapshot_until``
    seconds, the whole string's displacement is kept (for animation).
    """
    if string.tension is None:
        raise ValueError(f"The simulation needs the {string.note} string's tension")
    c, sigma0, sigma1, kappa = _coefficients(string, damping, stiffness)
    length = string.length
    k = 1 / sample_rate
    h_min = min_spacing(c, k, sigma1, kappa)

    max_intervals = int(np.floor(length / h_min))
    if n_intervals is None:
        n_intervals = max_intervals
    elif n_intervals > max_intervals:
        raise ValueError(
            f"{n_intervals} intervals is unstable at this time step; at most {max_intervals}"
        )
    h = length / n_intervals
    if sigma1 == 0 and kappa == 0:
        k = h / c  # exactly at the Courant limit: the scheme is then exact
    lam2 = (c * k / h) ** 2
    mu2 = (kappa * k / h**2) ** 2
    loss = 2 * sigma1 * k / h**2
    denominator = 1 + sigma0 * k

    x = np.linspace(0, length, n_intervals + 1)
    tension = string.tension
    density = tension / c**2

    def d2(y: np.ndarray) -> np.ndarray:
        """Second difference, zero at the fixed ends (also y_xx = 0 there)."""
        out = np.zeros_like(y)
        out[1:-1] = y[2:] - 2 * y[1:-1] + y[:-2]
        return out

    y_prev = _initial_displacement(x, string, pluck)
    d2_prev = d2(y_prev)
    # First step from rest: y(k) = y(0) + k^2/2 y_tt(0)
    y = y_prev + 0.5 * (lam2 * d2_prev - mu2 * d2(d2_prev))
    y[0] = y[-1] = 0.0

    n_steps = round(duration / k)
    force = np.empty(n_steps)
    energy = np.empty(n_steps) if track_energy else None
    snapshot_limit = snapshot_until if snapshot_until is not None else duration
    snapshots, snapshot_times = [], []

    # Bridge force: tension times the slope at x = 0. The grid displacement is
    # exactly a sum of discrete sine modes, so the slope is taken spectrally:
    # sum over modes of a_n n pi / L (a one-sided difference overstates the
    # upper harmonics on a coarse grid). The upper half of the grid's modes is
    # tapered (raised cosine) so the force steps don't ring; everything below
    # half the grid's range is untouched.
    n = np.arange(1, n_intervals)
    i = np.arange(n_intervals + 1)
    taper = np.clip((n - n_intervals / 2) / (n_intervals / 2), 0, 1)
    taper = 0.5 + 0.5 * np.cos(np.pi * taper)
    slope_weights = (
        (2 / n_intervals)
        * (taper * n * np.pi / length)
        @ np.sin(np.outer(n, i) * np.pi / n_intervals)
    )

    def bridge_force(y: np.ndarray) -> float:
        return tension * float(slope_weights @ y)

    for step in range(n_steps):
        if snapshot_every and step % snapshot_every == 0 and step * k <= snapshot_limit:
            snapshots.append(y_prev.copy())
            snapshot_times.append(step * k)
        force[step] = bridge_force(y_prev)

        d2_y = d2(y)
        y_next = (
            2 * y
            - (1 - sigma0 * k) * y_prev
            + lam2 * d2_y
            - mu2 * d2(d2_y)
            + loss * (d2_y - d2_prev)
        ) / denominator
        y_next[0] = y_next[-1] = 0.0

        if energy is not None:
            energy[step] = _energy(y_prev, y, k, h, tension, density, kappa, d2_prev, d2_y)
        y_prev, y, d2_prev = y, y_next, d2_y

    return FdtdResult(
        string=string,
        pluck=pluck,
        x=x,
        dt=k,
        bridge_force=force,
        snapshot_times=np.array(snapshot_times),
        snapshots=np.array(snapshots).reshape(len(snapshots), x.size),
        energy=energy,
        courant=c * k / h,
    )


def _energy(y0, y1, k, h, tension, density, kappa, d2_0, d2_1) -> float:
    """The scheme's conserved energy between steps n and n+1 (exactly constant without loss)."""
    kinetic = 0.5 * density * h * np.sum(((y1 - y0) / k) ** 2)
    potential = 0.5 * tension * h * np.sum(np.diff(y1) * np.diff(y0)) / h**2
    bending = 0.5 * density * kappa**2 * h * np.sum(d2_1 * d2_0) / h**4
    return float(kinetic + potential + bending)


def harmonic_amplitudes(result: FdtdResult, freqs: np.ndarray) -> np.ndarray:
    """Amplitude of the bridge force at each frequency in ``freqs``, Hann-windowed.

    Evaluates the windowed Fourier sum at exactly those frequencies, so a steady
    partial of amplitude A reads A (to within leakage from its neighbours).
    """
    force = result.bridge_force
    window = np.hanning(force.size)
    t = result.times
    phases = np.exp(-2j * np.pi * np.outer(freqs, t))
    return 2 * np.abs(phases @ (window * force)) / window.sum()
