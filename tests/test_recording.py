import shutil

import numpy as np
import pytest

from spectrum_modeler.audio import read_wav, write_wav
from spectrum_modeler.recording import (
    PRE_ROLL,
    ImportReport,
    ManifestRow,
    Session,
    clipped_runs,
    decode,
    detect_onsets,
    import_session,
    probe,
    read_manifest,
    write_manifest,
)
from spectrum_modeler.synthetic import load_truth


def clips_by_impairment(session):
    return {c["impairment"]: name for name, c in load_truth(session)["clips"].items()}


def test_manifest_round_trip(tmp_path):
    rows = [
        ManifestRow(file="ROOMTONE_2026-10-04.m4a", session="2026-10-04"),
        ManifestRow(
            file="E2_040mm_flesh.m4a",
            string=6,
            note="E2",
            pluck_mm=40.0,
            finger="flesh",
            stroke="free",
            plucks=3,
            session="2026-10-04",
            recorder="iphone_voice_memos",
            mic_distance_cm=40.0,
            notes="pluck 2 weak",
        ),
    ]
    path = tmp_path / "manifest.csv"
    write_manifest(path, rows)
    assert read_manifest(path) == rows
    assert rows[0].is_room_tone and not rows[1].is_room_tone
    assert rows[1].stem == "E2_040mm_flesh"


def test_clipped_runs_counts_runs_of_three_or_more():
    audio = np.array([0, 1, 1, 1, 0, -1, -1, 0, 1, 1, 1, 1, 0.5])
    assert clipped_runs(audio) == 2


def test_probe_and_decode_wav(tmp_path):
    path = tmp_path / "tone.wav"
    tone = 0.5 * np.sin(2 * np.pi * 440 * np.arange(4800) / 48_000)
    write_wav(path, tone, 48_000)
    info = probe(path)
    assert (info.codec, info.sample_rate, info.channels) == ("pcm_f32le", 48_000, 1)
    assert not info.lossy
    np.testing.assert_allclose(decode(path), tone, atol=1e-6)


def test_onsets_are_within_5_ms(synthetic_session):
    session, report = synthetic_session
    truth = load_truth(session)
    for clip in report.clips:
        expected = truth["clips"][clip.file]["onsets_s"]
        assert len(clip.onsets_s) == len(expected), clip.file
        np.testing.assert_allclose(clip.onsets_s, expected, atol=0.005)


def test_onset_detection_needs_a_real_rise():
    rng = np.random.default_rng(0)
    assert detect_onsets(rng.normal(0, 0.01, 48_000), 48_000).size == 0
    assert detect_onsets(np.zeros(48_000), 48_000).size == 0


def test_every_impairment_is_flagged(synthetic_session):
    session, report = synthetic_session
    clips = clips_by_impairment(session)

    def messages(file, severity):
        return [i.message for i in report.issues_for(file) if i.severity == severity]

    assert any("clipping" in m for m in messages(clips["clipping"], "warning"))
    assert any("lossy codec (aac)" in m for m in messages(clips["lossy"], "warning"))
    assert any("no manifest row" in m for m in messages(clips["missing-row"], "error"))
    # sympathetic peaks are flagged by measurement (test_measure.py)


def test_clean_clips_have_no_issues(synthetic_session):
    session, report = synthetic_session
    clips = clips_by_impairment(session)
    for clean in (clips[None],):
        assert report.issues_for(clean) == []
    assert not [i for i in report.issues if i.file == clips["sympathetic"]]


def test_takes_are_written_with_pre_roll(synthetic_session):
    session, report = synthetic_session
    for clip in report.clips:
        assert len(clip.takes) == 3
        for i, take in enumerate(clip.takes):
            assert (session.processed / take).is_file()
            assert clip.onset_in_take(i) == pytest.approx(PRE_ROLL, abs=1e-6)


def test_report_round_trips_through_json(synthetic_session):
    session, report = synthetic_session
    loaded = ImportReport.load(session.report)
    assert loaded.clips == report.clips
    assert loaded.issues == report.issues


def test_manifest_row_without_file_is_an_error(synthetic_session, tmp_path):
    session, _ = synthetic_session
    copy = Session(tmp_path / "copy")
    shutil.copytree(session.raw, copy.raw)
    rows = read_manifest(session.manifest)
    write_manifest(copy.manifest, [*rows, ManifestRow(file="E4_040mm_flesh.m4a", note="E4")])
    report = import_session(copy)
    assert any(
        i.file == "E4_040mm_flesh.m4a" and i.severity == "error" and "missing" in i.message
        for i in report.issues
    )


def test_decoded_clip_is_cached(synthetic_session):
    session, report = synthetic_session
    cached = session.processed / f"{report.clips[0].stem}.wav"
    audio, sr = read_wav(cached)
    assert sr == 48_000 and audio.size > 0


def test_missing_manifest_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError):
        import_session(Session(tmp_path))


def test_iphone_lossless_format_imports_cleanly(synthetic_session, tmp_path):
    """Voice Memos 'Lossless' is ALAC in .m4a, possibly stereo at 44.1 kHz."""
    import subprocess

    source_session, _ = synthetic_session
    session = Session(tmp_path)
    session.raw.mkdir()
    for src, dst in (("G3_130mm_flesh.wav", "G3_130mm_flesh.m4a"),
                     ("ROOMTONE_synthetic.wav", "ROOMTONE_x.m4a")):  # fmt: skip
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", str(source_session.raw / src),
             "-ac", "2", "-ar", "44100", "-c:a", "alac", str(session.raw / dst)],
            check=True,
        )  # fmt: skip
    write_manifest(session.manifest, [
        ManifestRow(file="ROOMTONE_x.m4a"),
        ManifestRow(file="G3_130mm_flesh.m4a", note="G3", pluck_mm=130.0, finger="flesh", plucks=3),
    ])  # fmt: skip
    report = import_session(session)
    assert report.issues == []
    (clip,) = report.clips
    assert clip.codec == "alac"
    assert len(clip.takes) == 3


def test_planned_rows_follow_the_recording_guide():
    from spectrum_modeler.recording import planned_rows

    rows = planned_rows(["E2", "G3"], "2026-10-04", mic_distance_cm=40, scale_length_mm=652)
    assert rows[0].file == "ROOMTONE_2026-10-04.m4a" and rows[0].is_room_tone
    names = [r.file for r in rows[1:]]
    assert names[:5] == [
        "E2_040mm_flesh.m4a", "E2_080mm_flesh.m4a", "E2_130mm_flesh.m4a",
        "E2_200mm_flesh.m4a", "E2_325mm_flesh.m4a",
    ]  # fmt: skip
    twelfth = rows[5]
    assert twelfth.pluck_mm == 326.0  # measured half-length, though named 325
    assert all(r.plucks == 3 and r.mic_distance_cm == 40 for r in rows[1:])
    assert len(rows) == 11


def test_reimport_replaces_old_takes_only(synthetic_session, tmp_path):
    source, report = synthetic_session
    session = Session(tmp_path)
    shutil.copytree(source.raw, session.raw)
    shutil.copy(source.manifest, session.manifest)
    session.processed.mkdir()
    stem = report.clips[0].stem
    stale = session.processed / f"{stem}_t9.wav"  # from an earlier import with more plucks
    unrelated = session.processed / f"{stem}_tasto.wav"
    for path in (stale, unrelated):
        write_wav(path, np.zeros(10), 48_000)
    import_session(session)
    assert not stale.exists()
    assert unrelated.exists()
