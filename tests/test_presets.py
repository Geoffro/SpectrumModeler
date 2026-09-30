import pytest

from spectrum_modeler.params import ModelOptions, PluckParams
from spectrum_modeler.presets import OPEN_STRING_NOTES, open_string, open_strings, pluck

# Nominal frequencies from design §3.2.
EXPECTED_F0 = {"E2": 82.41, "A2": 110.00, "D3": 146.83, "G3": 196.00, "B3": 246.94, "E4": 329.63}


def test_open_strings_are_lowest_first_with_standard_tuning():
    strings = open_strings()
    assert [s.note for s in strings] == list(OPEN_STRING_NOTES)
    for s in strings:
        assert s.f0 == pytest.approx(EXPECTED_F0[s.note], abs=0.005)
        assert s.tension is not None


def test_open_string_rejects_unknown_note():
    with pytest.raises(ValueError, match="Unknown open string"):
        open_string("C4")


def test_twelfth_fret_is_half_the_scale_length():
    assert pluck("twelfth_fret").distance_mm == pytest.approx(325.0)
    assert pluck("twelfth_fret", length=0.651).distance_mm == pytest.approx(325.5)


def test_pluck_accepts_distance_and_finger():
    p = pluck(123.0, finger="flesh")
    assert p.distance_mm == 123.0
    assert p.width_mm > 0


def test_pluck_rejects_unknown_names():
    with pytest.raises(ValueError, match="Unknown pluck preset"):
        pluck("middle")
    with pytest.raises(ValueError, match="Unknown finger"):
        pluck("tasto", finger="pick")


def test_beta_must_be_on_the_string():
    assert PluckParams(distance_mm=130).beta(0.650) == pytest.approx(0.2)
    for distance in (0, 650, 700):
        with pytest.raises(ValueError):
            PluckParams(distance_mm=distance).beta(0.650)


def test_model_options_validation():
    with pytest.raises(ValueError):
        ModelOptions(n_harmonics=0)
    with pytest.raises(ValueError):
        ModelOptions(observable="pressure")
