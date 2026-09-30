"""Presets: open strings, pluck positions, and finger types (design §3)."""

from pathlib import Path

from .params import PluckParams, StringParams
from .string_sets import DEFAULT_STRING_SET, StringSet, load_string_set

DEFAULT_SCALE_LENGTH = 0.650  # m


def _equal_tempered(semitones_from_a4: int) -> float:
    return 440.0 * 2 ** (semitones_from_a4 / 12)


# note -> (string number, f0 in Hz), lowest string first
OPEN_STRINGS: dict[str, tuple[int, float]] = {
    "E2": (6, _equal_tempered(-29)),
    "A2": (5, _equal_tempered(-24)),
    "D3": (4, _equal_tempered(-19)),
    "G3": (3, _equal_tempered(-14)),
    "B3": (2, _equal_tempered(-10)),
    "E4": (1, _equal_tempered(-5)),
}
OPEN_STRING_NOTES = tuple(OPEN_STRINGS)

# Distances from the saddle. The 12th fret is half the scale length, so it is
# computed from the actual length rather than stored.
PLUCK_POSITIONS_MM = {"ponticello": 40.0, "normal": 130.0, "tasto": 250.0}
TWELFTH_FRET = "twelfth_fret"
PLUCK_PRESETS = (*PLUCK_POSITIONS_MM, TWELFTH_FRET)

FINGER_WIDTHS_MM = {"ideal": 0.0, "nail": 1.5, "flesh": 10.0}


def open_string(
    note: str,
    *,
    string_set: str | Path | StringSet = DEFAULT_STRING_SET,
    length: float = DEFAULT_SCALE_LENGTH,
) -> StringParams:
    """Parameters for one open string, with tension etc. from the string set."""
    if note not in OPEN_STRINGS:
        raise ValueError(f"Unknown open string {note!r}; expected one of {OPEN_STRING_NOTES}")
    if not isinstance(string_set, StringSet):
        string_set = load_string_set(string_set)

    spec = string_set[note]
    return StringParams(
        note=note,
        f0=OPEN_STRINGS[note][1],
        length=length,
        tension=spec.tension,
        diameter=spec.diameter,
        youngs_modulus=spec.youngs_modulus,
        inharmonicity_b=spec.inharmonicity_b,
    )


def open_strings(
    *,
    string_set: str | Path | StringSet = DEFAULT_STRING_SET,
    length: float = DEFAULT_SCALE_LENGTH,
) -> list[StringParams]:
    """All six open strings, lowest first."""
    if not isinstance(string_set, StringSet):
        string_set = load_string_set(string_set)
    return [open_string(n, string_set=string_set, length=length) for n in OPEN_STRING_NOTES]


def pluck_position_mm(position: str | float, length: float = DEFAULT_SCALE_LENGTH) -> float:
    """Resolve a preset name or a distance in mm to a distance from the saddle in mm."""
    if isinstance(position, str):
        if position == TWELFTH_FRET:
            return length * 1000 / 2
        if position not in PLUCK_POSITIONS_MM:
            raise ValueError(f"Unknown pluck preset {position!r}; expected one of {PLUCK_PRESETS}")
        return PLUCK_POSITIONS_MM[position]
    return float(position)


def pluck(
    position: str | float,
    *,
    length: float = DEFAULT_SCALE_LENGTH,
    finger: str = "ideal",
    amplitude: float = 0.002,
) -> PluckParams:
    """Pluck parameters from a preset name (or distance in mm) and a finger type."""
    if finger not in FINGER_WIDTHS_MM:
        raise ValueError(f"Unknown finger {finger!r}; expected one of {tuple(FINGER_WIDTHS_MM)}")
    return PluckParams(
        distance_mm=pluck_position_mm(position, length),
        amplitude=amplitude,
        width_mm=FINGER_WIDTHS_MM[finger],
        finger=finger,
    )
