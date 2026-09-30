"""Import a recording session: decode, check against the manifest, split into takes.

Reads <session>/manifest.csv and <session>/raw/, writes <session>/processed/
(decoded clips, one WAV per take, and import_report.json). Exits with status 1
if there are errors.

Examples:
    python scripts/import_recordings.py                              # recordings/
    python scripts/import_recordings.py --session output/synthetic_session
"""

import argparse
import sys
from pathlib import Path

from spectrum_modeler.recording import Session, import_session


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--session", type=Path, default=Path("recordings"))
    args = parser.parse_args()

    try:
        report = import_session(Session(args.session))
    except (FileNotFoundError, RuntimeError) as e:
        sys.exit(f"error: {e}")

    print(f"{'clip':28} {'codec':10} {'plucks':>6} {'peak dBFS':>10} {'floor dBFS':>11}")
    for clip in report.clips:
        print(
            f"{clip.file:28} {clip.codec:10} {len(clip.takes):>6} "
            f"{clip.peak_dbfs:>10.1f} {clip.floor_dbfs:>11.1f}"
        )
    if report.room_tone_dbfs is not None:
        print(f"room tone: {report.room_tone_dbfs:.1f} dBFS")

    print()
    for issue in report.issues:
        print(issue)
    print(
        f"\n{len(report.clips)} clips imported, {len(report.errors)} errors, "
        f"{len(report.warnings)} warnings. Report: {Session(args.session).report}"
    )
    if report.errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
