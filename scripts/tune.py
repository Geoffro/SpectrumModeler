"""Open the tuner: compare a guitar profile with each measured take and adjust it by hand.

Needs a session that has been imported and analyzed, and a profile from calibrate.py.
Adjustments are saved to the profile's manual section (press Save).

Example:
    python scripts/tune.py --session recordings --profile my_guitar
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

from spectrum_modeler.measure import load_session_measurements
from spectrum_modeler.profile import PROFILE_DIR, load_profile, profile_path
from spectrum_modeler.recording import Session
from spectrum_modeler.tuning import TunerState
from spectrum_modeler.viz.tuner import TunerApp


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--session", type=Path, default=Path("recordings"))
    parser.add_argument("--profile", required=True, help="profile name or .json path")
    parser.add_argument("--profiles", type=Path, default=PROFILE_DIR, help="profile directory")
    args = parser.parse_args()

    session = Session(args.session)
    try:
        path = profile_path(args.profile, args.profiles)
        state = TunerState(load_profile(path), load_session_measurements(session), path, session)
    except (FileNotFoundError, ValueError) as e:
        sys.exit(f"error: {e}")

    app = TunerApp(state)
    plt.show()
    if state.dirty:
        print("closed with unsaved changes")
    del app


if __name__ == "__main__":
    main()
