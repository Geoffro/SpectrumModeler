"""Measure plucks: every take in an imported session, or any single audio file.

Session mode (after import_recordings.py) writes <session>/measurements/<clip>.json
and plots to <session>/plots/. Ad-hoc mode measures the first pluck in any file
ffmpeg can read and writes a plot to --out.

Examples:
    python scripts/analyze.py output/synthetic_session
    python scripts/analyze.py recordings --plots all
    python scripts/analyze.py ~/Downloads/some_note.mp3               # f0 detected
    python scripts/analyze.py my_E2.m4a --note E2 --pluck-mm 130      # with model overlay
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from spectrum_modeler.audio import read_wav
from spectrum_modeler.measure import (
    Measurement,
    measure,
    measure_clip,
    model_for,
    save_measurements,
)
from spectrum_modeler.presets import FINGER_WIDTHS_MM, OPEN_STRING_NOTES, OPEN_STRINGS
from spectrum_modeler.recording import (
    SAMPLE_RATE,
    ImportReport,
    Session,
    decode,
    detect_onsets,
)
from spectrum_modeler.viz.measured import plot_measurement


def summary_row(m: Measurement) -> str:
    b = f"{m.inharmonicity_b:.1e}" if m.inharmonicity_b is not None else "-"
    s0 = f"{m.sigma0:.2f}" if m.sigma0 is not None else "-"
    s2 = f"{m.sigma2:.1e}" if m.sigma2 is not None else "-"
    off = ", ".join(f"{f:.0f}" for f in m.off_series_hz) or "-"
    name = f"{m.clip} t{m.take}" if m.take is not None else m.clip
    return (
        f"{name:30} {m.f0:8.2f} {b:>8} {s0:>6} {s2:>8} "
        f"{m.n_reliable:>4}/{m.harmonics.size:<4} {off}"
    )


HEADER = f"{'take':30} {'f0 Hz':>8} {'B':>8} {'σ0':>6} {'σ2':>8} {'harmonics':>9} off-series Hz"


def save_plot(audio, sr, onset, m: Measurement, path: Path) -> None:
    fig = plot_measurement(audio, sr, onset, m, model_for(m))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def analyze_session(root: Path, plots: str) -> None:
    session = Session(root)
    if not session.report.is_file():
        sys.exit(f"error: no import report in {session.processed}; run import_recordings.py first")
    report = ImportReport.load(session.report)

    print(HEADER)
    flagged = []
    for clip in report.clips:
        measurements = measure_clip(session, clip)
        if len(measurements) < len(clip.takes):
            print(
                f"{clip.stem}: {len(clip.takes) - len(measurements)} take(s) could not be measured"
            )
        for m in measurements:
            print(summary_row(m))
            if m.off_series_hz.size:
                flagged.append(m)
            if plots == "all" or (plots == "first" and m is measurements[0]):
                audio, sr = read_wav(session.processed / f"{clip.stem}_t{m.take}.wav")
                onset = round(m.onset * sr)
                save_plot(audio, sr, onset, m, session.plots / f"{clip.stem}_t{m.take}.png")
        save_measurements(session.measurements / f"{clip.stem}.json", measurements)

    for m in flagged:
        freqs = ", ".join(f"{f:.0f} Hz" for f in m.off_series_hz)
        print(f"WARNING {m.clip} t{m.take}: peaks off the harmonic series at {freqs} "
              "(sympathetic strings or distortion?)")  # fmt: skip
    print(f"\nmeasurements: {session.measurements}")
    if plots != "none":
        print(f"plots: {session.plots}")


def analyze_file(path: Path, args: argparse.Namespace) -> None:
    try:
        audio = decode(path)
        onsets = detect_onsets(audio, SAMPLE_RATE)
        if onsets.size == 0:
            raise ValueError("no pluck onset found")
        f0_hint = args.f0 or (OPEN_STRINGS[args.note][1] if args.note else None)
        m = measure(
            audio, SAMPLE_RATE, int(onsets[0]),
            f0_hint=f0_hint, clip=path.stem, note=args.note,
            pluck_mm=args.pluck_mm, finger=args.finger,
        )  # fmt: skip
    except (ValueError, RuntimeError) as e:
        sys.exit(f"error: could not measure {path.name}: {e}")

    if onsets.size > 1:
        print(
            f"{onsets.size} plucks found; measuring the first, at {onsets[0] / SAMPLE_RATE:.2f} s"
        )
    print(HEADER)
    print(summary_row(m))
    out = args.out / f"{path.stem}.png"
    save_plot(audio, SAMPLE_RATE, int(onsets[0]), m, out)
    print(f"plot: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("target", type=Path, help="session directory or audio file")
    parser.add_argument("--plots", choices=["first", "all", "none"], default="first",
                        help="session mode: which takes to plot")  # fmt: skip
    adhoc = parser.add_argument_group("ad-hoc mode")
    adhoc.add_argument("--note", choices=OPEN_STRING_NOTES, help="open string, as f0 hint")
    adhoc.add_argument("--f0", type=float, help="f0 hint in Hz")
    adhoc.add_argument("--pluck-mm", type=float, help="pluck distance, for the model overlay")
    adhoc.add_argument("--finger", choices=list(FINGER_WIDTHS_MM), default="flesh")
    adhoc.add_argument("--out", type=Path, default=Path("output/adhoc"))
    args = parser.parse_args()

    if args.target.is_dir():
        analyze_session(args.target, args.plots)
    elif args.target.is_file():
        analyze_file(args.target, args)
    else:
        sys.exit(f"error: {args.target} does not exist")


if __name__ == "__main__":
    main()
