import pytest

from spectrum_modeler.presets import OPEN_STRING_NOTES, open_strings
from spectrum_modeler.string_sets import (
    DEFAULT_STRING_SET,
    LBF_TO_N,
    available_string_sets,
    load_string_set,
)

# Linear density in g/m for a 650 mm scale, from the table in design §3.2.
EXPECTED_MU_G_PER_M = {"E2": 5.89, "A2": 3.70, "D3": 1.98, "G3": 0.84, "B3": 0.54, "E4": 0.41}


def test_default_set_is_bundled():
    assert DEFAULT_STRING_SET in available_string_sets()


def test_ej46_has_all_open_strings_with_published_tensions():
    ej46 = load_string_set("daddario_ej46")
    assert set(ej46.strings) == set(OPEN_STRING_NOTES)
    assert ej46["E2"].item == "J4606"
    assert ej46["E2"].tension == pytest.approx(15.2 * LBF_TO_N)
    assert ej46["A2"].tension == pytest.approx(75.62, abs=0.01)


def test_derived_linear_density_matches_design_table():
    for string in open_strings():
        assert string.linear_density * 1000 == pytest.approx(
            EXPECTED_MU_G_PER_M[string.note], abs=0.006
        )


def test_derived_linear_density_reproduces_f0():
    for string in open_strings(length=0.655):
        wave_speed = (string.tension / string.linear_density) ** 0.5
        assert wave_speed / (2 * string.length) == pytest.approx(string.f0)


def test_load_from_path_accepts_newtons_and_optional_dimensions(tmp_path):
    path = tmp_path / "custom.toml"
    path.write_text(
        'name = "Custom"\n'
        "[strings.E4]\n"
        "tension_n = 70.0\n"
        "diameter_mm = 0.72\n"
        "youngs_modulus_gpa = 5.0\n"
    )
    custom = load_string_set(path)
    assert custom.key == "custom"
    assert custom["E4"].tension == 70.0
    assert custom["E4"].diameter == pytest.approx(0.00072)
    assert custom["E4"].youngs_modulus == pytest.approx(5e9)


@pytest.mark.parametrize(
    "entry",
    ["tension_n = 70.0\ntension_lbf = 15.0\n", "item = 'X'\n", "tension_n = -1.0\n"],
    ids=["both-units", "no-tension", "negative"],
)
def test_invalid_tension_is_rejected(tmp_path, entry):
    path = tmp_path / "bad.toml"
    path.write_text("[strings.E4]\n" + entry)
    with pytest.raises(ValueError):
        load_string_set(path)


def test_unknown_set_and_missing_string_give_clear_errors():
    with pytest.raises(ValueError, match="Unknown string set"):
        load_string_set("no_such_set")
    with pytest.raises(KeyError, match="no C4 string"):
        load_string_set()["C4"]
