"""Fit a guitar profile to a session's measurements.

Reads <session>/measurements/ (from analyze.py) and writes calibration/<name>.json
plus a report plot in <session>/plots/. Refitting an existing profile keeps its
manual adjustments from the tuner.

Examples:
    python scripts/calibrate.py --session recordings --name my_guitar
    python scripts/calibrate.py --session output/synthetic_session --name synthetic
"""

import argparse
import collections
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from spectrum_modeler.calibrate import MAX_BETA_SHIFT_MM, calibrate
from spectrum_modeler.measure import load_session_measurements
from spectrum_modeler.presets import DEFAULT_SCALE_LENGTH
from spectrum_modeler.profile import PROFILE_DIR, load_profile, profile_path
from spectrum_modeler.recording import Session, read_manifest
from spectrum_modeler.string_sets import DEFAULT_STRING_SET
from spectrum_modeler.synthetic import TRUTH_FILE, load_truth, truth_response_db
from spectrum_modeler.viz.calibration import GOOD_FIT_DB, plot_calibration_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--session", type=Path, default=Path("recordings"))
    parser.add_argument("--name", required=True, help="profile name")
    parser.add_argument("--profiles", type=Path, default=PROFILE_DIR, help="profile directory")
    parser.add_argument("--string-set", default=DEFAULT_STRING_SET)
    parser.add_argument(
        "--scale-length", type=float, default=DEFAULT_SCALE_LENGTH * 1000, help="mm"
    )
    parser.add_argument("--no-refine-position", action="store_true",
                        help="use manifest pluck positions as given")  # fmt: skip
    args = parser.parse_args()

    session = Session(args.session)
    try:
        measurements = load_session_measurements(session)
    except FileNotFoundError as e:
        sys.exit(f"error: {e}")
    recorders = collections.Counter(r.recorder for r in read_manifest(session.manifest))
    recorder = recorders.most_common(1)[0][0] if recorders else None

    result = calibrate(
        measurements,
        name=args.name,
        scale_length=args.scale_length / 1000,
        string_set=args.string_set,
        recorder=recorder,
        refine_beta=not args.no_refine_position,
    )
    profile = result.profile
    path = profile_path(args.name, args.profiles)
    if path.is_file():
        previous = load_profile(path)
        profile = profile.with_manual_from(previous)
        for note, manual in profile.manual.items():
            print(f"kept manual adjustments for {note}: {manual.describe()}")
        for finger, width in profile.manual_finger_width_mm.items():
            print(f"kept manual finger width for {finger}: {width:g} mm")
    profile.save(path)

    is_synthetic = (session.root / TRUTH_FILE).is_file()
    truth = load_truth(session) if is_synthetic else None
    print(
        f"\n{'string':6} {'f0 Hz':>8} {'B':>9} {'σ0':>6} {'σ2':>9}"
        + ("   truth: B, σ0, σ2" if truth else "")
    )
    for note, fit in profile.strings.items():
        b = f"{fit.inharmonicity_b:.2e}" if fit.inharmonicity_b is not None else "-"
        line = f"{note:6} {fit.f0:8.2f} {b:>9} {fit.sigma0:6.2f} {fit.sigma2:9.2e}"
        if truth:
            t = truth["strings"][note]
            line += f"   {t['b']:.2e}, {t['sigma0']:.2f}, {t['sigma2']:.2e}"
        print(line)
    widths = ", ".join(f"{f} {w:.1f} mm" for f, w in profile.finger_width_mm.items())
    print(
        f"finger width: {widths}"
        + (f" (truth {truth['config']['finger_width_mm']} mm)" if truth else "")
    )

    rms = np.array([t.rms_db for t in result.takes])
    print(
        f"fit: {result.rms_db:.2f} dB RMS overall; "
        f"{np.mean(rms <= GOOD_FIT_DB) * 100:.0f}% of {len(rms)} takes within {GOOD_FIT_DB:g} dB"
    )
    for take in result.takes:
        if abs(take.fitted_mm - take.manifest_mm) >= MAX_BETA_SHIFT_MM - 0.05:
            print(f"WARNING {take.clip} t{take.take}: fitted position hit the "
                  f"±{MAX_BETA_SHIFT_MM:g} mm limit; check the tape mark")  # fmt: skip

    fig = plot_calibration_report(result, truth_response_db if is_synthetic else None)
    plot_path = session.plots / f"calibration_{args.name}.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path)
    plt.close(fig)
    print(f"\nprofile: {path}\nreport: {plot_path}")


if __name__ == "__main__":
    main()
