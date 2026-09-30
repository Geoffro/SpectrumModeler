import dataclasses

import numpy as np
import pytest
from scipy.integrate import quad

from spectrum_modeler import body
from spectrum_modeler.modal import (
    coefficients,
    displacement_coefficients,
    envelope_nulls,
    finger_width_factor,
    initial_shape,
    missing_harmonics,
    mode_contributions,
)
from spectrum_modeler.params import ModelOptions, PluckParams, StringParams
from spectrum_modeler.presets import PLUCK_PRESETS, open_string, pluck

E2 = open_string("E2")
H = 0.002


@pytest.mark.parametrize("preset", PLUCK_PRESETS)
def test_mode_sum_reconstructs_triangle(preset):
    p = pluck(preset)
    x = np.linspace(0, E2.length, 2001)
    reconstructed = mode_contributions(x, E2, p, n_modes=2000).sum(axis=0)
    error = np.abs(reconstructed - initial_shape(x, E2.length, p)).max()
    assert error < 0.005 * H


@pytest.mark.parametrize("preset", PLUCK_PRESETS)
def test_coefficients_match_numerical_sine_transform(preset):
    p = pluck(preset)
    L = E2.length
    xp = p.beta(L) * L
    n = np.arange(1, 21)
    numerical = [
        2
        / L
        * quad(
            lambda x, k=k: initial_shape(x, L, p) * np.sin(k * np.pi * x / L),
            0,
            L,
            points=[xp],
        )[0]
        for k in n
    ]
    np.testing.assert_allclose(
        displacement_coefficients(n, p.beta(L), H), numerical, rtol=1e-6, atol=1e-12
    )


def test_twelfth_fret_removes_every_even_harmonic():
    spectrum = coefficients(E2, pluck("twelfth_fret"))
    even, odd = spectrum.amps[1::2], spectrum.amps[0::2]
    assert np.all(even == 0.0)
    assert np.all(odd != 0.0)


def test_missing_harmonics_are_multiples_of_one_over_beta():
    np.testing.assert_array_equal(missing_harmonics(0.5, 10), [2, 4, 6, 8, 10])
    np.testing.assert_array_equal(missing_harmonics(0.2, 20), [5, 10, 15, 20])
    # 40 mm on 650 mm is beta = 4/65: the first node harmonic is 65.
    assert missing_harmonics(40 / 650, 40).size == 0


def test_envelope_nulls_need_not_be_whole_numbers():
    np.testing.assert_allclose(envelope_nulls(0.3, 10), [1 / 0.3, 2 / 0.3, 3 / 0.3])
    np.testing.assert_allclose(envelope_nulls(0.5, 6), [2, 4, 6])


@pytest.mark.parametrize("beta", [0.06, 0.2, 0.38])
def test_mirror_symmetry(beta):
    n = np.arange(1, 41)
    np.testing.assert_allclose(
        displacement_coefficients(n, beta, H),
        (-1) ** (n + 1) * displacement_coefficients(n, 1 - beta, H),
        atol=1e-15,
    )


@pytest.mark.parametrize("preset", ["ponticello", "normal", "tasto"])
def test_observables_roll_off_as_expected(preset):
    p = pluck(preset)
    beta = p.beta(E2.length)
    opts = {
        obs: ModelOptions(n_harmonics=40, observable=obs)
        for obs in ("displacement", "velocity", "bridge_force")
    }
    disp = coefficients(E2, p, opts["displacement"])
    vel = coefficients(E2, p, opts["velocity"])
    force = coefficients(E2, p, opts["bridge_force"])
    n = disp.harmonics

    # Displacement envelope falls as 1/n^2; bridge force as 1/n.
    envelope = 2 * H / (np.pi**2 * beta * (1 - beta))
    assert np.all(np.abs(disp.amps) * n**2 <= envelope * (1 + 1e-12))
    force_envelope = E2.tension * np.pi / E2.length * envelope
    assert np.all(np.abs(force.amps) * n <= force_envelope * (1 + 1e-12))

    # Exact relations between observables.
    np.testing.assert_allclose(vel.amps, 2 * np.pi * disp.freqs * disp.amps)
    np.testing.assert_allclose(force.amps, E2.tension * disp.amps * n * np.pi / E2.length)


def test_spectrum_basics():
    spectrum = coefficients(E2, pluck("normal"), ModelOptions(n_harmonics=12, damping=False))
    np.testing.assert_allclose(spectrum.freqs, np.arange(1, 13) * E2.f0)
    assert np.all(spectrum.decay_rates == 0)
    assert spectrum.beta == pytest.approx(0.2)
    assert spectrum.units == "N"
    levels = spectrum.levels_db(floor_db=-80)
    assert levels.max() == 0
    assert levels[4] == -80  # harmonic 5 is missing at beta = 0.2


def test_bridge_force_needs_tension():
    bare = StringParams(note="E2", f0=82.41)
    with pytest.raises(ValueError, match="tension"):
        coefficients(bare, PluckParams(distance_mm=130))
    coefficients(bare, PluckParams(distance_mm=130), ModelOptions(observable="displacement"))


# --- Realism layers (design §2.3) ---


def test_damping_layer_follows_sigma_formula():
    spectrum = coefficients(E2, pluck("normal"))
    np.testing.assert_allclose(spectrum.decay_rates, E2.sigma0 + E2.sigma2 * spectrum.freqs**2)
    assert np.all(np.diff(spectrum.decay_rates) > 0)  # higher harmonics die faster


def test_finger_width_is_a_gentle_low_pass():
    ideal = coefficients(E2, pluck("normal"), ModelOptions(finger_width=False))
    flesh = coefficients(E2, pluck("normal", finger="flesh"))
    nonzero = ideal.amps != 0
    ratio = flesh.amps[nonzero] / ideal.amps[nonzero]
    assert ratio[0] == pytest.approx(1, abs=1e-3)
    assert np.all(np.diff(ratio) < 0)
    assert np.all((ratio > 0) & (ratio < 1))


def test_finger_width_factor_zero_where_finger_spans_a_wavelength():
    # sinc(n w / 2L) = 0 when the finger covers one full wavelength of mode n
    length, n = 0.65, 10
    assert finger_width_factor(np.array([n]), 2 * length / n, length)[0] == pytest.approx(
        0, abs=1e-15
    )


def test_inharmonicity_raises_upper_partials():
    stiff = dataclasses.replace(E2, inharmonicity_b=1e-4)
    spectrum = coefficients(stiff, pluck("normal"), ModelOptions(inharmonicity=True))
    n = spectrum.harmonics
    np.testing.assert_allclose(spectrum.freqs, n * E2.f0 * np.sqrt(1 + 1e-4 * n**2))
    with pytest.raises(ValueError, match="needs B"):
        coefficients(E2, pluck("normal"), ModelOptions(inharmonicity=True))


def test_b_coefficient_from_dimensions():
    s = StringParams(note="E4", f0=329.63, tension=74.7, diameter=0.72e-3, youngs_modulus=5e9)
    expected = np.pi**3 * 5e9 * 0.72e-3**4 / (64 * 74.7 * 0.65**2)
    assert s.b_coefficient == pytest.approx(expected)
    assert dataclasses.replace(s, inharmonicity_b=3e-5).b_coefficient == 3e-5
    assert E2.b_coefficient is None


def test_body_layer_boosts_partials_near_resonances():
    plain = coefficients(E2, pluck("normal"))
    colored = coefficients(E2, pluck("normal"), ModelOptions(body=True))
    nonzero = plain.amps != 0
    gain_db = 20 * np.log10(colored.amps[nonzero] / plain.amps[nonzero])
    np.testing.assert_allclose(gain_db, body.response_db(plain.freqs[nonzero]))
    # E2's harmonic 1 (82 Hz) and 2 (165 Hz) sit near the 100 and 200 Hz resonances
    assert gain_db[0] > 2 and gain_db[1] > 2
    assert abs(gain_db[-1]) < 1  # far above the resonances: little change


def test_body_response_peaks_at_resonance_frequencies():
    for r in body.ILLUSTRATIVE_RESONANCES:
        assert body.response_db([r.freq])[0] >= r.gain_db - 1
