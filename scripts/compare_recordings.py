"""Compare two recordings: spectrograms side by side, their difference, and harmonic views.

Each recording is aligned at its pluck and matched in loudness. Accepts any format
ffmpeg reads, including a session's takes (recordings/processed/<clip>_t1.wav).
Writes <A>_vs_<B>.png and an A-then-B WAV to --out, and prints a summary.

Examples:
    python scripts/compare_recordings.py a.m4a b.m4a
    python scripts/compare_recordings.py recordings/raw/E2_040mm_flesh.m4a \\
        recordings/raw/E2_200mm_flesh.m4a --note E2 --pluck-b 2
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from spectrum_modeler.analysis import spectral_centroid
from spectrum_modeler.audio import sequence, write_wav
from spectrum_modeler.compare import MAX_DURATION, compare_files
from spectrum_modeler.presets import OPEN_STRING_NOTES, OPEN_STRINGS
from spectrum_modeler.recording import SAMPLE_RATE
from spectrum_modeler.viz.compare import plot_comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("a", type=Path)
    parser.add_argument("b", type=Path)
    parser.add_argument("--pluck-a", type=int, default=1, help="which pluck in A (1 = first)")
    parser.add_argument("--pluck-b", type=int, default=1, help="which pluck in B")
    hint = parser.add_mutually_exclusive_group()
    hint.add_argument("--note", choices=OPEN_STRING_NOTES, help="f0 hint for both")
    hint.add_argument("--f0", type=float, help="f0 hint for both, Hz")
    parser.add_argument("--duration", type=float, default=MAX_DURATION, help="seconds, at most")
    parser.add_argument("--max-freq", type=float, default=5000.0, help="spectrogram top, Hz")
    parser.add_argument("--harmonics", type=int, nargs="+", default=[1, 2, 3, 4],
                        help="up to four harmonics whose decay to plot")  # fmt: skip
    parser.add_argument("--out", type=Path, default=Path("output/compare"))
    args = parser.parse_args()

    f0 = args.f0 or (OPEN_STRINGS[args.note][1] if args.note else None)
    try:
        comparison = compare_files(
            args.a, args.b,
            plucks=(args.pluck_a, args.pluck_b),
            f0_hints=(f0, f0),
            max_duration=args.duration,
        )  # fmt: skip
    except (ValueError, RuntimeError) as e:
        sys.exit(f"error: {e}")

    a, b = comparison.a, comparison.b
    stem = f"{a.name}_vs_{b.name}"
    args.out.mkdir(parents=True, exist_ok=True)
    fig = plot_comparison(comparison, max_freq=args.max_freq, harmonics=tuple(args.harmonics[:4]))
    fig.savefig(args.out / f"{stem}.png")
    plt.close(fig)
    write_wav(args.out / f"{stem}.wav", sequence([a.audio, b.audio], SAMPLE_RATE), SAMPLE_RATE)

    print(f"compared the first {comparison.duration:.2f} s after each pluck, matched in loudness")
    for label, rec in (("A", a), ("B", b)):
        m = rec.measurement
        if m is None:
            print(f"{label} {rec.name}: not measured ({rec.problem})")
            continue
        ok = m.reliable & np.isfinite(m.initial_db)
        centroid = spectral_centroid(m.freqs[ok], 10 ** (m.initial_db[ok] / 20))
        decay = f"σ0 {m.sigma0:.2f}/s, σ2 {m.sigma2:.1e}" if m.sigma0 is not None else "decay -"
        print(f"{label} {rec.name}: f0 {m.f0:.2f} Hz, brightness {centroid:.0f} Hz, "
              f"{m.n_reliable} harmonics, {decay}")  # fmt: skip

    if a.measurement is not None and b.measurement is not None:
        print("\nharmonic   B − A at the pluck (dB)")
        ma, mb = a.measurement, b.measurement
        for n in range(1, min(ma.harmonics.size, mb.harmonics.size, 12) + 1):
            both = ma.reliable[n - 1] and mb.reliable[n - 1]
            diff = mb.initial_db[n - 1] - ma.initial_db[n - 1] if both else np.nan
            print(f"{n:>8}   " + (f"{diff:+6.1f}" if np.isfinite(diff) else "     -"))
    print(f"\nplot: {args.out / stem}.png\naudio (A then B): {args.out / stem}.wav")


if __name__ == "__main__":
    main()
