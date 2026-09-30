"""Open the explorer: drag the pluck point and watch and hear the overtones change.

Optional: --profile adds a calibrated mode (your guitar as recorded); --session
lets you play the recorded take nearest the current pluck position.

Examples:
    python scripts/explore.py
    python scripts/explore.py --profile my_guitar --session recordings
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

from spectrum_modeler.explorer import ExplorerState
from spectrum_modeler.measure import load_session_measurements
from spectrum_modeler.profile import PROFILE_DIR, load_profile
from spectrum_modeler.recording import Session
from spectrum_modeler.string_sets import DEFAULT_STRING_SET
from spectrum_modeler.viz.explorer import ExplorerApp


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", help="guitar profile name or .json path")
    parser.add_argument("--profiles", type=Path, default=PROFILE_DIR, help="profile directory")
    parser.add_argument("--session", type=Path, help="analyzed session, for recordings")
    parser.add_argument("--string-set", default=DEFAULT_STRING_SET)
    args = parser.parse_args()

    try:
        profile = load_profile(args.profile, args.profiles) if args.profile else None
        session = Session(args.session) if args.session else None
        measurements = load_session_measurements(session) if session else None
    except FileNotFoundError as e:
        sys.exit(f"error: {e}")

    state = ExplorerState(
        string_set=profile.string_set if profile else args.string_set,
        profile=profile,
        session=session,
        measurements=measurements,
    )
    app = ExplorerApp(state)
    plt.show()
    del app


if __name__ == "__main__":
    main()
