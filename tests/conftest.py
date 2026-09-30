import matplotlib
import pytest

matplotlib.use("Agg")

from spectrum_modeler.recording import import_session
from spectrum_modeler.synthetic import IMPAIRMENTS, SyntheticConfig, generate_session

# Six clips: E2 and G3 at 40 mm, 130 mm, and the 12th fret. Clips 1-4 each
# carry one impairment, in IMPAIRMENTS order; clips 0 and 5 are clean.
SMALL_SESSION = SyntheticConfig(
    notes=("E2", "G3"),
    positions=(40.0, 130.0, "twelfth_fret"),
    ring=1.5,
    gap=0.5,
    impairments=IMPAIRMENTS,
)


@pytest.fixture(scope="session")
def synthetic_session(tmp_path_factory):
    """A small imported synthetic session: (session, import report)."""
    session = generate_session(tmp_path_factory.mktemp("session"), SMALL_SESSION)
    return session, import_session(session)


# Three strings at all five positions, clean: enough spread for calibration.
CALIBRATION_SESSION = SyntheticConfig(notes=("E2", "D3", "E4"), ring=1.5, gap=0.5, seed=1)


@pytest.fixture(scope="session")
def calibration_measurements(tmp_path_factory):
    """(session, measurements of every take) for a clean synthetic session."""
    from spectrum_modeler.measure import measure_clip

    session = generate_session(tmp_path_factory.mktemp("calibration"), CALIBRATION_SESSION)
    report = import_session(session)
    return session, [m for clip in report.clips for m in measure_clip(session, clip)]
