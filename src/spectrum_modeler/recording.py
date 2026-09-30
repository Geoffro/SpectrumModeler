"""Recording import: manifest, decoding, pluck segmentation, and quality checks.

A session directory holds ``manifest.csv`` and ``raw/`` (design §4.2). Import
decodes each clip to 48 kHz mono float WAV in ``processed/``, splits it into one
take per pluck, and writes ``processed/import_report.json`` with every issue
found. Decoding uses ffmpeg, so any format it reads is accepted.
"""

import csv
import json
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from .audio import read_wav, rms, write_wav
from .presets import OPEN_STRINGS, TWELFTH_FRET

SAMPLE_RATE = 48_000
AUDIO_SUFFIXES = {".m4a", ".wav", ".flac", ".aiff", ".aif", ".mp3"}
LOSSY_CODECS = {"aac", "mp3", "mp2", "opus", "vorbis", "ac3"}
REPORT_FILE = "import_report.json"

PRE_ROLL = 0.02  # s of audio kept before each onset in a take
LOW_LEVEL_DBFS = -30.0  # peak below this is flagged
FLOOR_TOLERANCE_DB = 3.0  # clip floor this far above room tone is flagged
MIN_SNR_DB = 40.0  # peak-to-floor below this is flagged

MANIFEST_FIELDS = (
    "file",
    "string",
    "note",
    "pluck_mm",
    "finger",
    "stroke",
    "plucks",
    "session",
    "recorder",
    "mic_distance_cm",
    "notes",
)


# --- Manifest -----------------------------------------------------------------


@dataclass(frozen=True)
class ManifestRow:
    file: str
    string: int | None = None
    note: str | None = None
    pluck_mm: float | None = None
    finger: str | None = None
    stroke: str | None = None
    plucks: int = 0
    session: str | None = None
    recorder: str | None = None
    mic_distance_cm: float | None = None
    notes: str = ""

    @property
    def stem(self) -> str:
        return Path(self.file).stem

    @property
    def is_room_tone(self) -> bool:
        return self.file.upper().startswith("ROOMTONE")


def _optional(value: str | None, cast):
    value = (value or "").strip()
    return cast(value) if value else None


def read_manifest(path: str | Path) -> list[ManifestRow]:
    with open(path, newline="") as f:
        return [
            ManifestRow(
                file=r["file"].strip(),
                string=_optional(r.get("string"), int),
                note=_optional(r.get("note"), str),
                pluck_mm=_optional(r.get("pluck_mm"), float),
                finger=_optional(r.get("finger"), str),
                stroke=_optional(r.get("stroke"), str),
                plucks=_optional(r.get("plucks"), int) or 0,
                session=_optional(r.get("session"), str),
                recorder=_optional(r.get("recorder"), str),
                mic_distance_cm=_optional(r.get("mic_distance_cm"), float),
                notes=(r.get("notes") or "").strip(),
            )
            for r in csv.DictReader(f)
        ]


def write_manifest(path: str | Path, rows: list[ManifestRow]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: "" if v is None else v for k, v in asdict(row).items()})


# Pluck positions from the recording guide, mm from the saddle; the 12th fret is
# half the measured scale length but is named 325 in file names.
GUIDE_POSITIONS: tuple[float | str, ...] = (40.0, 80.0, 130.0, 200.0, TWELFTH_FRET)
TWELFTH_FRET_NAME_MM = 325


def planned_rows(
    notes: list[str],
    date: str,
    *,
    positions: tuple[float | str, ...] = GUIDE_POSITIONS,
    finger: str = "flesh",
    stroke: str = "free",
    plucks: int = 3,
    recorder: str = "iphone_voice_memos",
    mic_distance_cm: float | None = None,
    scale_length_mm: float = 650.0,
) -> list[ManifestRow]:
    """Manifest rows for a planned session: a room-tone clip, then every string and position.

    File names follow the recording guide (``E2_040mm_flesh.m4a``), so renaming each
    memo to match is all that is left to do.
    """
    common = dict(session=date, recorder=recorder, mic_distance_cm=mic_distance_cm)
    rows = [ManifestRow(file=f"ROOMTONE_{date}.m4a", **common)]
    for note in notes:
        for position in positions:
            if position == TWELFTH_FRET:
                name_mm, pluck_mm = TWELFTH_FRET_NAME_MM, scale_length_mm / 2
            else:
                name_mm = pluck_mm = float(position)
            rows.append(
                ManifestRow(
                    file=f"{note}_{round(name_mm):03d}mm_{finger}.m4a",
                    string=OPEN_STRINGS[note][0],
                    note=note,
                    pluck_mm=pluck_mm,
                    finger=finger,
                    stroke=stroke,
                    plucks=plucks,
                    **common,
                )
            )
    return rows


# --- Session layout -----------------------------------------------------------


@dataclass(frozen=True)
class Session:
    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.csv"

    @property
    def raw(self) -> Path:
        return self.root / "raw"

    @property
    def processed(self) -> Path:
        return self.root / "processed"

    @property
    def measurements(self) -> Path:
        return self.root / "measurements"

    @property
    def plots(self) -> Path:
        return self.root / "plots"

    @property
    def report(self) -> Path:
        return self.processed / REPORT_FILE


# --- Decoding -----------------------------------------------------------------


@dataclass(frozen=True)
class AudioInfo:
    codec: str
    sample_rate: int
    channels: int
    duration: float  # s

    @property
    def lossy(self) -> bool:
        return self.codec in LOSSY_CODECS


def _run(tool: str, *args: str) -> subprocess.CompletedProcess:
    if shutil.which(tool) is None:
        raise RuntimeError(f"{tool} not found; it is installed by environment.yml")
    try:
        return subprocess.run([tool, *args], capture_output=True, check=True)
    except subprocess.CalledProcessError as e:
        raise ValueError(f"{tool} failed on {args[-1]}: {e.stderr.decode().strip()}") from None


def probe(path: str | Path) -> AudioInfo:
    """Codec, sample rate, channels, and duration of the first audio stream."""
    result = _run(
        "ffprobe",
        "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=codec_name,sample_rate,channels:format=duration",
        "-of", "json",
        str(path),
    )  # fmt: skip
    data = json.loads(result.stdout)
    if not data.get("streams"):
        raise ValueError(f"No audio stream in {path}")
    stream = data["streams"][0]
    return AudioInfo(
        codec=stream["codec_name"],
        sample_rate=int(stream["sample_rate"]),
        channels=int(stream["channels"]),
        duration=float(data["format"]["duration"]),
    )


def decode(path: str | Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Any file ffmpeg can read, as mono float64 at ``sample_rate``."""
    result = _run(
        "ffmpeg",
        "-v", "error",
        "-i", str(path),
        "-ac", "1",
        "-ar", str(sample_rate),
        "-f", "f32le",
        "-",
    )  # fmt: skip
    return np.frombuffer(result.stdout, dtype="<f4").astype(np.float64)


def load_clip(session: Session, row: ManifestRow, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Decoded clip, cached in ``processed/`` until the raw file changes."""
    src = session.raw / row.file
    cached = session.processed / f"{row.stem}.wav"
    if cached.is_file() and cached.stat().st_mtime >= src.stat().st_mtime:
        audio, sr = read_wav(cached)
        if sr == sample_rate:
            return audio
    audio = decode(src, sample_rate)
    write_wav(cached, audio, sample_rate, check_range=False)
    return audio


# --- Levels, onsets, clipping --------------------------------------------------


def dbfs(value: float) -> float:
    return float(20 * np.log10(max(value, 1e-12)))


def frame_levels_db(audio: np.ndarray, sample_rate: int, frame: float, hop: float) -> np.ndarray:
    """RMS level in dBFS of each ``frame``-second window, every ``hop`` seconds."""
    n_frame, n_hop = round(frame * sample_rate), round(hop * sample_rate)
    if audio.size < n_frame:
        return np.array([dbfs(rms(audio))]) if audio.size else np.array([-240.0])
    windows = np.lib.stride_tricks.sliding_window_view(audio, n_frame)[::n_hop]
    return 20 * np.log10(np.sqrt(np.mean(windows**2, axis=1)) + 1e-12)


def noise_floor_dbfs(audio: np.ndarray, sample_rate: int) -> float:
    """Level of the quietest 10% of 50 ms frames: the floor between plucks."""
    return float(np.percentile(frame_levels_db(audio, sample_rate, 0.05, 0.025), 10))


def detect_onsets(
    audio: np.ndarray,
    sample_rate: int,
    *,
    rise_db: float = 20.0,
    rearm_db: float = 10.0,
    min_gap: float = 0.3,
) -> np.ndarray:
    """Sample indices where plucks start.

    A pluck is detected when the 5 ms level jumps ``rise_db`` above both the
    quietest level of the preceding 50 ms and the clip's noise floor. Detection
    re-arms only once the level falls back within ``rearm_db`` of the floor, so
    each pluck must decay or be damped before the next (as the recording guide
    asks). The onset is refined to the first sample ``rise_db`` above the
    preceding quiet level.
    """
    n_frame, n_hop = round(0.005 * sample_rate), round(0.001 * sample_rate)
    levels = frame_levels_db(audio, sample_rate, 0.005, 0.001)
    # 1st percentile: a single long pluck may leave only a sliver of silence.
    floor = np.percentile(levels, 1)
    lookback = 50  # frames = 50 ms
    # Before the file starts counts as silence, so a pluck at time 0 is found.
    padded = np.concatenate([np.full(lookback, -240.0), levels])
    recent_min = np.lib.stride_tricks.sliding_window_view(padded, lookback + 1).min(axis=1)

    onsets: list[int] = []
    armed = True
    for i, level in enumerate(levels):
        if armed and level > floor + rise_db and level > recent_min[i] + rise_db:
            start = i * n_hop
            threshold = 10 ** ((recent_min[i] + rise_db) / 20)
            above = np.flatnonzero(np.abs(audio[start : start + n_frame]) > threshold)
            onset = start + (int(above[0]) if above.size else 0)
            if not onsets or onset - onsets[-1] >= min_gap * sample_rate:
                onsets.append(onset)
            armed = False
        elif not armed and level < floor + rearm_db:
            armed = True
    return np.array(onsets, dtype=int)


def clipped_runs(audio: np.ndarray, threshold: float = 0.999, min_run: int = 3) -> int:
    """Number of runs of at least ``min_run`` consecutive samples at full scale."""
    at_limit = np.concatenate([[False], np.abs(audio) >= threshold, [False]])
    edges = np.flatnonzero(np.diff(at_limit.astype(int)))
    starts, ends = edges[::2], edges[1::2]
    return int(np.sum(ends - starts >= min_run))


# --- Import ---------------------------------------------------------------------

Severity = Literal["error", "warning"]


@dataclass(frozen=True)
class Issue:
    file: str
    severity: Severity
    message: str

    def __str__(self) -> str:
        return f"{self.severity.upper():7} {self.file or '(session)'}: {self.message}"


@dataclass
class ClipReport:
    file: str
    stem: str
    note: str | None
    pluck_mm: float | None
    finger: str | None
    codec: str
    peak_dbfs: float
    floor_dbfs: float
    onsets_s: list[float] = field(default_factory=list)  # in the clip
    takes: list[str] = field(default_factory=list)  # file names in processed/
    take_starts_s: list[float] = field(default_factory=list)  # in the clip

    def onset_in_take(self, i: int) -> float:
        """Seconds from the start of take ``i`` (0-based) to its pluck."""
        return self.onsets_s[i] - self.take_starts_s[i]


@dataclass
class ImportReport:
    session: str
    room_tone_dbfs: float | None
    clips: list[ClipReport]
    issues: list[Issue]

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "warning"]

    def issues_for(self, file: str) -> list[Issue]:
        return [i for i in self.issues if i.file == file]

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "ImportReport":
        data = json.loads(Path(path).read_text())
        return cls(
            session=data["session"],
            room_tone_dbfs=data["room_tone_dbfs"],
            clips=[ClipReport(**c) for c in data["clips"]],
            issues=[Issue(**i) for i in data["issues"]],
        )


def import_session(session: Session, sample_rate: int = SAMPLE_RATE) -> ImportReport:
    """Decode, check, and split every clip in the session; save and return the report."""
    if not session.manifest.is_file():
        raise FileNotFoundError(f"No manifest at {session.manifest}")
    rows = read_manifest(session.manifest)
    issues: list[Issue] = []

    on_disk = (
        {p.name for p in session.raw.iterdir() if p.suffix.lower() in AUDIO_SUFFIXES}
        if session.raw.is_dir()
        else set()
    )
    listed = {r.file for r in rows}
    for name in sorted(on_disk - listed):
        issues.append(Issue(name, "error", "in raw/ but has no manifest row; not imported"))
    for name in sorted(listed - on_disk):
        issues.append(Issue(name, "error", "listed in the manifest but missing from raw/"))
    present = [r for r in rows if r.file in on_disk]
    session.processed.mkdir(parents=True, exist_ok=True)

    room_levels = [dbfs(rms(load_clip(session, r, sample_rate))) for r in present if r.is_room_tone]
    room_tone = float(np.mean(room_levels)) if room_levels else None
    if room_tone is None:
        issues.append(Issue("", "warning", "no room-tone clip, so noise floors can't be checked"))

    clips = [
        _import_clip(session, row, room_tone, sample_rate, issues)
        for row in present
        if not row.is_room_tone
    ]
    report = ImportReport(str(session.root), room_tone, clips, issues)
    report.save(session.report)
    return report


def _import_clip(
    session: Session,
    row: ManifestRow,
    room_tone: float | None,
    sample_rate: int,
    issues: list[Issue],
) -> ClipReport:
    def flag(severity: Severity, message: str) -> None:
        issues.append(Issue(row.file, severity, message))

    info = probe(session.raw / row.file)
    if info.lossy:
        flag("warning", f"lossy codec ({info.codec}); record lossless (Voice Memos: Lossless)")

    audio = load_clip(session, row, sample_rate)
    peak = dbfs(np.abs(audio).max(initial=0))
    floor = noise_floor_dbfs(audio, sample_rate)
    if runs := clipped_runs(audio):
        flag("warning", f"clipping: {runs} run(s) at full scale")
    if peak < LOW_LEVEL_DBFS:
        flag("warning", f"low level: peak {peak:.1f} dBFS")
    if room_tone is not None and floor > room_tone + FLOOR_TOLERANCE_DB:
        flag(
            "warning",
            f"noise floor {floor:.1f} dBFS is {floor - room_tone:.1f} dB above room tone "
            "(automatic gain or noise?)",
        )
    if peak - floor < MIN_SNR_DB:
        flag("warning", f"only {peak - floor:.0f} dB between peak and noise floor")

    onsets = detect_onsets(audio, sample_rate)
    if onsets.size == 0:
        flag("error", "no plucks detected")
    elif row.plucks and onsets.size != row.plucks:
        flag("warning", f"{onsets.size} plucks detected; manifest says {row.plucks}")

    for old in session.processed.glob(f"{row.stem}_t[0-9]*.wav"):
        old.unlink()
    pre = round(PRE_ROLL * sample_rate)
    takes, starts = [], []
    for i, onset in enumerate(onsets):
        start = max(0, onset - pre)
        end = onsets[i + 1] - pre if i + 1 < onsets.size else audio.size
        name = f"{row.stem}_t{i + 1}.wav"
        write_wav(session.processed / name, audio[start:end], sample_rate, check_range=False)
        takes.append(name)
        starts.append(start / sample_rate)

    return ClipReport(
        file=row.file,
        stem=row.stem,
        note=row.note,
        pluck_mm=row.pluck_mm,
        finger=row.finger,
        codec=info.codec,
        peak_dbfs=peak,
        floor_dbfs=floor,
        onsets_s=[o / sample_rate for o in onsets],
        takes=takes,
        take_starts_s=starts,
    )
