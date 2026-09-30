import pytest

from spectrum_modeler.synthetic import SyntheticConfig, generate_session


def test_config_rejects_unknown_or_too_many_impairments():
    with pytest.raises(ValueError, match="Unknown impairments"):
        SyntheticConfig(impairments=("wow",))
    with pytest.raises(ValueError, match="more clips"):
        SyntheticConfig(notes=("E2",), positions=(40.0,), impairments=("lossy",))


def test_existing_session_is_replaced_only_on_request(tmp_path):
    (tmp_path / "keep.txt").write_text("x")
    config = SyntheticConfig(notes=("E2",), positions=(130.0,), plucks=1, ring=0.5)
    with pytest.raises(FileExistsError):
        generate_session(tmp_path, config)
    session = generate_session(tmp_path, config, overwrite=True)
    assert not (tmp_path / "keep.txt").exists()
    assert sorted(p.name for p in session.raw.iterdir()) == [
        "E2_130mm_flesh.wav",
        "ROOMTONE_synthetic.wav",
    ]
