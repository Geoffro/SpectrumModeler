import dataclasses
import os

import numpy as np
import pytest

from spectrum_modeler.modal import coefficients
from spectrum_modeler.params import ModelOptions
from spectrum_modeler.presets import open_string, pluck
from spectrum_modeler.profile import (
    GuitarProfile,
    StringFit,
    StringManual,
    load_profile,
    profile_path,
)


def make_profile(**overrides) -> GuitarProfile:
    values = dict(
        name="test",
        scale_length_mm=652.0,
        string_set="daddario_ej46",
        recorder="synthetic",
        created="2026-10-05",
        source_measurements=["E2_040mm_flesh_t1"],
        response_freqs=[80.0, 400.0, 4000.0],
        response_db=[-6.0, 3.0, -2.0],
        finger_width_mm={"flesh": 12.0},
        strings={
            "E2": StringFit(
                f0=82.3,
                inharmonicity_b=5e-5,
                sigma0=0.9,
                sigma2=1.5e-6,
                sigma_n=[0.95, None, 1.2],
            )
        },
    )
    values.update(overrides)
    return GuitarProfile(**values)


def test_json_round_trip_keeps_fitted_and_manual(tmp_path):
    profile = make_profile(
        manual={"E2": StringManual(sigma0=1.1, offsets_db={7: -2.0})},
        manual_finger_width_mm={"flesh": 9.0},
    )
    path = tmp_path / "p.json"
    profile.save(path)
    assert load_profile(path) == profile


def test_empty_manual_sections_are_dropped(tmp_path):
    profile = make_profile(manual={"E2": StringManual()})
    profile.save(tmp_path / "p.json")
    assert load_profile(tmp_path / "p.json").manual == {}


def test_manual_values_override_fitted():
    profile = make_profile()
    assert profile.damping("E2") == (0.9, 1.5e-6)
    profile.manual_for("E2").sigma0 = 1.4
    assert profile.damping("E2") == (1.4, 1.5e-6)
    assert profile.strings["E2"].sigma0 == 0.9  # fitted value untouched
    profile.manual_finger_width_mm["flesh"] = 8.0
    assert profile.finger_width("flesh") == 8.0
    assert profile.finger_width("nail") is None


def test_decay_rates_prefer_measured_until_damping_is_adjusted():
    profile = make_profile()
    n = np.array([1, 2, 3])
    freqs = n * 82.3
    formula = 0.9 + 1.5e-6 * freqs**2
    np.testing.assert_allclose(profile.decay_rates("E2", n, freqs), [0.95, formula[1], 1.2])
    assert profile.uses_measured_decay("E2") and not profile.uses_measured_decay("G3")
    profile.manual_for("E2").sigma2 = 3e-6
    assert not profile.uses_measured_decay("E2")
    np.testing.assert_allclose(profile.decay_rates("E2", n, freqs), 0.9 + 3e-6 * freqs**2)


def test_response_is_interpolated_in_log_frequency_and_held_at_the_ends():
    profile = make_profile()
    np.testing.assert_allclose(profile.response_at([20.0, 80.0, 400.0, 10_000.0]), [-6, -6, 3, -2])
    midpoint = np.sqrt(80.0 * 400.0)  # halfway in log frequency
    assert profile.response_at([midpoint])[0] == pytest.approx(-1.5)


def test_refit_keeps_manual_adjustments():
    old = make_profile(manual={"E2": StringManual(offsets_db={3: 1.5})},
                       manual_finger_width_mm={"flesh": 9.0})  # fmt: skip
    refit = make_profile(finger_width_mm={"flesh": 13.0}).with_manual_from(old)
    assert refit.finger_width_mm == {"flesh": 13.0}
    assert refit.finger_width("flesh") == 9.0
    assert refit.manual["E2"].offsets_db == {3: 1.5}
    refit.manual["E2"].offsets_db[4] = 1.0
    assert 4 not in old.manual["E2"].offsets_db  # a copy, not shared


def test_load_returns_independent_copies_and_sees_file_changes(tmp_path):
    path = tmp_path / "p.json"
    make_profile().save(path)
    first = load_profile(path)
    first.manual_for("E2").sigma0 = 5.0
    assert load_profile(path).manual == {}

    make_profile(name="changed").save(path)
    stat = path.stat()
    os.utime(path, (stat.st_atime, stat.st_mtime + 10))
    assert load_profile(path).name == "changed"


def test_profile_path_and_missing_profile(tmp_path):
    assert profile_path("my_guitar", tmp_path) == tmp_path / "my_guitar.json"
    assert profile_path(tmp_path / "x.json") == tmp_path / "x.json"
    with pytest.raises(FileNotFoundError):
        load_profile("nope", tmp_path)


# --- Profiles in the model ---------------------------------------------------------------


def test_profile_switches_on_calibrated_layers():
    profile = make_profile(manual={"E2": StringManual(offsets_db={2: 6.0})})
    e2, p = open_string("E2"), pluck("normal", finger="flesh")
    plain = coefficients(e2, p, ModelOptions(n_harmonics=3))
    calibrated = coefficients(e2, p, ModelOptions(n_harmonics=3, profile=profile))

    assert calibrated.string.f0 == 82.3
    assert calibrated.string.length == pytest.approx(0.652)
    n = calibrated.harmonics
    np.testing.assert_allclose(calibrated.freqs, n * 82.3 * np.sqrt(1 + 5e-5 * n**2))
    assert calibrated.pluck.width_mm == 12.0
    assert calibrated.decay_rates[0] == 0.95  # measured sigma_n

    # Body response and the manual offset on harmonic 2 are applied on top of the pluck.
    reference = coefficients(
        dataclasses.replace(e2, f0=82.3, length=0.652, inharmonicity_b=5e-5),
        dataclasses.replace(p, width_mm=12.0),
        ModelOptions(n_harmonics=3, inharmonicity=True),
    )
    gain_db = 20 * np.log10(calibrated.amps / reference.amps)
    expected = profile.response_at(calibrated.freqs) + np.array([0, 6.0, 0])
    np.testing.assert_allclose(gain_db, expected)
    assert not np.allclose(plain.amps, calibrated.amps)


def test_profile_layers_can_be_switched_off_explicitly():
    profile = make_profile()
    e2, p = open_string("E2"), pluck("normal", finger="flesh")
    off = coefficients(e2, p, ModelOptions(profile=profile, body=False, inharmonicity=False))
    np.testing.assert_allclose(off.freqs, off.harmonics * 82.3)
    manual_width = coefficients(e2, p, ModelOptions(profile=profile, body=False))
    assert manual_width.pluck.width_mm == 12.0


def test_damping_modes_and_overrides_with_a_profile():
    e2, p = open_string("E2"), pluck("normal")

    def rates(**options):
        opts = ModelOptions(n_harmonics=3, profile=make_profile(), **options)
        return coefficients(e2, p, opts)

    measured = rates()
    assert measured.decay_rates[0] == 0.95 and measured.decay_rates[2] == 1.2
    np.testing.assert_allclose(rates(damping="flat").decay_rates, 0.9)  # the profile's sigma0
    adjusted = rates(sigma2=1e-6)  # the formula everywhere, with the profile's sigma0
    np.testing.assert_allclose(adjusted.decay_rates, 0.9 + 1e-6 * adjusted.freqs**2)
    np.testing.assert_allclose(rates(damping="flat", sigma0=2.0).decay_rates, 2.0)
    assert not rates(damping="off").decay_rates.any()


def test_profile_by_name(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    make_profile(name="mine").save(tmp_path / "calibration" / "mine.json")
    spectrum = coefficients(open_string("E2"), pluck("normal"), ModelOptions(profile="mine"))
    assert spectrum.string.f0 == 82.3


def test_strings_missing_from_the_profile_keep_defaults():
    profile = make_profile()
    g3 = coefficients(open_string("G3"), pluck("normal"), ModelOptions(profile=profile))
    assert g3.string.f0 == pytest.approx(196.0, abs=0.01)
    assert g3.string.length == pytest.approx(0.652)
