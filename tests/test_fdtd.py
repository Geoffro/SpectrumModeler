import dataclasses

import matplotlib.pyplot as plt
import numpy as np
import pytest

from spectrum_modeler.fdtd import harmonic_amplitudes, simulate
from spectrum_modeler.modal import coefficients
from spectrum_modeler.params import ModelOptions, StringParams
from spectrum_modeler.presets import open_string, pluck

STIFF_E2 = dataclasses.replace(open_string("E2"), inharmonicity_b=5e-5)


@pytest.mark.parametrize("note", ["E2", "E4"])
@pytest.mark.parametrize("position", ["ponticello", "normal", "twelfth_fret"])
@pytest.mark.parametrize("finger", ["ideal", "flesh"])
def test_bridge_force_matches_modal_model(note, position, finger):
    string, p = open_string(note), pluck(position, finger=finger)
    result = simulate(string, p, 0.3)
    modal = coefficients(string, p, ModelOptions(n_harmonics=20, damping=False))
    usable = np.abs(modal.amps) > 1e-2 * np.abs(modal.amps).max()  # within 40 dB
    simulated = harmonic_amplitudes(result, modal.freqs[usable])
    error_db = 20 * np.log10(simulated / np.abs(modal.amps[usable]))
    assert np.abs(error_db).max() < 1.0


def test_ideal_string_runs_exactly_at_the_courant_limit():
    result = simulate(open_string("G3"), pluck("normal"), 0.01)
    assert result.courant == pytest.approx(1.0)


@pytest.mark.parametrize("string", [open_string("E2"), STIFF_E2], ids=["ideal", "stiff"])
def test_energy_is_conserved_without_damping(string):
    result = simulate(string, pluck("normal"), 0.2, stiffness=string is STIFF_E2, track_energy=True)
    drift = np.abs(result.energy / result.energy[0] - 1).max()
    assert drift < 1e-10


def test_energy_only_falls_with_damping():
    result = simulate(STIFF_E2, pluck("ponticello"), 0.5, damping=True, stiffness=True,
                      track_energy=True)  # fmt: skip
    assert np.all(np.diff(result.energy) <= 1e-12 * result.energy[0])
    assert result.energy[-1] < 0.9 * result.energy[0]
    assert result.courant < 1  # stiffness and loss tighten the stability limit


def test_unstable_grid_is_refused():
    with pytest.raises(ValueError, match="unstable"):
        simulate(open_string("E2"), pluck("normal"), 0.01, n_intervals=400)


def test_stiffness_sharpens_upper_partials():
    result = simulate(STIFF_E2, pluck("ponticello"), 0.5, stiffness=True)
    n = 20
    stretched = n * STIFF_E2.f0 * np.sqrt(1 + 5e-5 * n**2)
    at_stretched, at_harmonic = harmonic_amplitudes(result, np.array([stretched, n * STIFF_E2.f0]))
    assert at_stretched > 10 * at_harmonic


def test_damping_decays_at_the_model_rate():
    string = open_string("G3")
    result = simulate(string, pluck("normal"), 1.0, damping=True)
    half = result.bridge_force.size // 2
    halves = [dataclasses.replace(result, bridge_force=part)
              for part in (result.bridge_force[:half], result.bridge_force[half:])]  # fmt: skip
    first, second = (harmonic_amplitudes(h, np.array([string.f0]))[0] for h in halves)
    expected = string.sigma0 + string.sigma2 * string.f0**2
    measured = np.log(first / second) / (half * result.dt)
    assert measured == pytest.approx(expected, rel=0.05)


def test_snapshots_for_animation():
    result = simulate(
        open_string("E4"), pluck("tasto"), 0.02, snapshot_every=4, snapshot_until=0.01
    )
    assert result.snapshots.shape == (result.snapshot_times.size, result.x.size)
    assert result.snapshot_times[-1] <= 0.01
    np.testing.assert_allclose(result.snapshots[:, [0, -1]], 0)


def test_needs_tension_and_b():
    with pytest.raises(ValueError, match="tension"):
        simulate(StringParams(note="E2", f0=82.4), pluck("normal"), 0.01)
    with pytest.raises(ValueError, match="needs B"):
        simulate(open_string("E2"), pluck("normal"), 0.01, stiffness=True)


def test_animation_and_plots_render(tmp_path):
    from spectrum_modeler.viz.animate import (
        animate_string,
        plot_filmstrip,
        plot_modal_check,
        save_animation,
    )

    string, p = open_string("E4"), pluck("ponticello")
    period = 1 / string.f0
    result = simulate(string, p, 0.1, snapshot_every=8, snapshot_until=period)
    path = save_animation(animate_string(result, slowdown=20, fps=10), tmp_path / "e4.gif", fps=10)
    assert path.stat().st_size > 0
    assert len(plot_filmstrip(result).axes) == 8
    plot_modal_check(result, coefficients(string, p, ModelOptions(n_harmonics=20, damping=False)))
    plt.close("all")
