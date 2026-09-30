"""Generate a synthetic recording session with known parameters, for testing the pipeline.

Examples:
    python scripts/make_synthetic_session.py                        # all strings, no impairments
    python scripts/make_synthetic_session.py --impair all --force   # every import check triggered
    python scripts/make_synthetic_session.py --notes E2 G3 --ring 2
"""

import argparse
from pathlib import Path

from spectrum_modeler.presets import OPEN_STRING_NOTES
from spectrum_modeler.synthetic import IMPAIRMENTS, TRUTH_FILE, SyntheticConfig, generate_session


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("output/synthetic_session"))
    parser.add_argument("--notes", nargs="+", choices=OPEN_STRING_NOTES, default=OPEN_STRING_NOTES)
    parser.add_argument(
        "--impair",
        nargs="+",
        choices=[*IMPAIRMENTS, "all"],
        default=[],
        help="problems to inject, one clip each",
    )
    parser.add_argument("--plucks", type=int, default=3)
    parser.add_argument("--ring", type=float, default=3.0, help="seconds each pluck rings")
    parser.add_argument("--noise", type=float, default=-70.0, help="noise level, dBFS RMS")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="replace an existing session")
    args = parser.parse_args()

    impairments = IMPAIRMENTS if "all" in args.impair else tuple(args.impair)
    config = SyntheticConfig(
        notes=tuple(args.notes),
        plucks=args.plucks,
        ring=args.ring,
        noise_dbfs=args.noise,
        impairments=impairments,
        seed=args.seed,
    )
    session = generate_session(args.out, config, overwrite=args.force)
    n_clips = len(list(session.raw.iterdir()))
    print(f"wrote {n_clips} clips (including room tone) to {session.raw}")
    print(f"truth in {session.root / TRUTH_FILE}")
    if impairments:
        print(f"impairments: {', '.join(impairments)}")
    print(f"next: python scripts/import_recordings.py --session {session.root}")


if __name__ == "__main__":
    main()
