"""Write the manifest for a recording session before you record, and list the file names.

Rename each Voice Memo to the listed name (without .m4a) as you go. Afterwards, fix
any row that changed: a different microphone distance, a note about a bad pluck.

Examples:
    python scripts/new_manifest.py --date 2026-10-04 --notes E2 G3 --mic-distance 40   # pilot
    python scripts/new_manifest.py --date 2026-10-11 --notes all --mic-distance 40 --append
"""

import argparse
import sys
from pathlib import Path

from spectrum_modeler.presets import FINGER_WIDTHS_MM, OPEN_STRING_NOTES
from spectrum_modeler.recording import Session, planned_rows, read_manifest, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--session", type=Path, default=Path("recordings"))
    parser.add_argument("--date", required=True, help="YYYY-MM-DD, used in the room-tone name")
    parser.add_argument("--notes", nargs="+", required=True, choices=["all", *OPEN_STRING_NOTES])
    parser.add_argument("--finger", default="flesh", choices=list(FINGER_WIDTHS_MM))
    parser.add_argument("--plucks", type=int, default=3)
    parser.add_argument("--mic-distance", type=float, help="cm, phone to soundhole")
    parser.add_argument("--scale-length", type=float, default=650.0, help="mm, as measured")
    parser.add_argument("--recorder", default="iphone_voice_memos")
    parser.add_argument("--append", action="store_true", help="add to an existing manifest")
    args = parser.parse_args()

    session = Session(args.session)
    notes = list(OPEN_STRING_NOTES) if "all" in args.notes else args.notes
    planned = planned_rows(
        notes,
        args.date,
        finger=args.finger,
        plucks=args.plucks,
        recorder=args.recorder,
        mic_distance_cm=args.mic_distance,
        scale_length_mm=args.scale_length,
    )

    existing = []
    if session.manifest.is_file():
        if not args.append:
            sys.exit(f"error: {session.manifest} exists; use --append to add to it")
        existing = read_manifest(session.manifest)
    known = {row.file for row in existing}
    new = [row for row in planned if row.file not in known]
    session.raw.mkdir(parents=True, exist_ok=True)
    write_manifest(session.manifest, existing + new)

    print(f"{len(new)} rows added to {session.manifest}; audio goes in {session.raw}/\n")
    print("Record and rename, in this order:")
    for row in new:
        what = "10 s of silence" if row.is_room_tone else f"{row.plucks} plucks"
        print(f"  [ ] {row.stem:24} {what}")
    skipped = len(planned) - len(new)
    if skipped:
        print(f"\n({skipped} planned clips were already in the manifest)")


if __name__ == "__main__":
    main()
