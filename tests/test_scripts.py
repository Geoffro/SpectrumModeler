"""Smoke tests: each command-line script runs end to end and writes what it says."""

import json
import runpy
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

from spectrum_modeler.audio import write_wav
from spectrum_modeler.presets import open_string, pluck
from spectrum_modeler.recording import Session, read_manifest
from spectrum_modeler.synthesis import render

SCRIPTS = Path(__file__).parent.parent / "scripts"
SR = 48_000


def run(script: str, *args, monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", [script, *map(str, args)])
    runpy.run_path(str(SCRIPTS / script), run_name="__main__")


def pluck_wav(path: Path, note: str = "G3", position="normal", lead: float = 0.2) -> Path:
    audio = 0.3 * render(open_string(note), pluck(position, finger="flesh"), duration=1.5)
    audio = np.concatenate([np.zeros(round(lead * SR)), audio])
    write_wav(path, audio + np.random.default_rng(0).normal(0, 1e-4, audio.size), SR)
    return path


@pytest.mark.slow
def test_synthetic_pipeline(tmp_path, monkeypatch, capsys):
    session = tmp_path / "session"
    profiles = tmp_path / "calibration"
    run("make_synthetic_session.py", "--out", session, "--notes", "E2", "G3", "--ring", 1.5,
        monkeypatch=monkeypatch)  # fmt: skip
    run("import_recordings.py", "--session", session, monkeypatch=monkeypatch)
    run("analyze.py", session, monkeypatch=monkeypatch)
    assert len(list((session / "measurements").glob("*.json"))) == 10
    assert len(list((session / "plots").glob("*.png"))) == 10  # first take of each clip

    run("calibrate.py", "--session", session, "--name", "syn", "--profiles", profiles,
        monkeypatch=monkeypatch)  # fmt: skip
    profile_file = profiles / "syn.json"
    assert (session / "plots" / "calibration_syn.png").is_file()
    assert "truth: B, σ0, σ2" in capsys.readouterr().out

    # A refit keeps manual adjustments and says so.
    data = json.loads(profile_file.read_text())
    data["strings"]["E2"]["manual"] = {"sigma0": 2.0}
    profile_file.write_text(json.dumps(data))
    run("calibrate.py", "--session", session, "--name", "syn", "--profiles", profiles,
        monkeypatch=monkeypatch)  # fmt: skip
    assert "kept manual adjustments for E2: sigma0=2" in capsys.readouterr().out
    assert json.loads(profile_file.read_text())["strings"]["E2"]["manual"] == {"sigma0": 2.0}


def test_import_exits_with_status_1_on_errors(synthetic_session, tmp_path, monkeypatch):
    source, _ = synthetic_session  # has a clip with no manifest row
    copy = Session(tmp_path / "copy")
    shutil.copytree(source.raw, copy.raw)
    shutil.copy(source.manifest, copy.manifest)
    with pytest.raises(SystemExit) as exit_info:
        run("import_recordings.py", "--session", copy.root, monkeypatch=monkeypatch)
    assert exit_info.value.code == 1
    assert copy.report.is_file()


def test_analyze_needs_an_imported_session(tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="no import report"):
        run("analyze.py", tmp_path, monkeypatch=monkeypatch)


def test_analyze_single_file(tmp_path, monkeypatch, capsys):
    path = pluck_wav(tmp_path / "g3.wav")
    run("analyze.py", path, "--note", "G3", "--pluck-mm", 130, "--out", tmp_path / "out",
        monkeypatch=monkeypatch)  # fmt: skip
    assert (tmp_path / "out" / "g3.png").is_file()
    row = capsys.readouterr().out.splitlines()[-2]
    assert row.startswith("g3") and " 196." in row


def test_analyze_unmeasurable_file_is_a_clear_error(tmp_path, monkeypatch):
    path = tmp_path / "silence.wav"
    write_wav(path, np.zeros(SR), SR)
    with pytest.raises(SystemExit, match=r"could not measure silence\.wav"):
        run("analyze.py", path, "--out", tmp_path, monkeypatch=monkeypatch)


def test_compare_positions(tmp_path, monkeypatch):
    run("compare_positions.py", "--string", "E2", "--a", 40, "--b", "twelfth_fret",
        "--duration", 0.5, "--detail", "--out", tmp_path, monkeypatch=monkeypatch)  # fmt: skip
    written = {p.name for p in tmp_path.iterdir()}
    assert written == {
        "E2_40mm_vs_twelfth_fret.png",
        "E2_40mm_vs_twelfth_fret_spectrogram.png",
        "E2_40mm.wav",
        "E2_twelfth_fret.wav",
        "E2_40mm_then_twelfth_fret.wav",
        "E2_40mm_modes.png",
        "E2_twelfth_fret_modes.png",
    }


def test_compare_recordings(tmp_path, monkeypatch, capsys):
    a = pluck_wav(tmp_path / "a.wav", position="tasto")
    b = pluck_wav(tmp_path / "b.wav", position="ponticello")
    run("compare_recordings.py", a, b, "--note", "G3", "--out", tmp_path / "out",
        monkeypatch=monkeypatch)  # fmt: skip
    assert (tmp_path / "out" / "a_vs_b.png").is_file()
    assert (tmp_path / "out" / "a_vs_b.wav").is_file()
    assert "B − A at the pluck" in capsys.readouterr().out


def test_new_manifest_writes_then_appends(tmp_path, monkeypatch, capsys):
    args = ("--session", tmp_path, "--date", "2026-10-04", "--mic-distance", 40)
    run("new_manifest.py", *args, "--notes", "E2", monkeypatch=monkeypatch)
    assert len(read_manifest(tmp_path / "manifest.csv")) == 6  # room tone + 5 positions
    assert (tmp_path / "raw").is_dir()

    with pytest.raises(SystemExit, match="--append"):
        run("new_manifest.py", *args, "--notes", "E2", monkeypatch=monkeypatch)

    capsys.readouterr()
    run("new_manifest.py", *args, "--notes", "E2", "G3", "--append", monkeypatch=monkeypatch)
    assert "5 rows added" in capsys.readouterr().out
    assert len(read_manifest(tmp_path / "manifest.csv")) == 11
