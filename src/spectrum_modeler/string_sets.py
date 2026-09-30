"""String sets: published tensions, and optional dimensions, for a set of strings.

Bundled sets live in ``spectrum_modeler/data/string_sets/<key>.toml``. A set can
also be loaded from any TOML file path. Tension may be given as ``tension_lbf`` or
``tension_n``; it is always stored in newtons. Optional per string:
``diameter_mm`` and ``youngs_modulus_gpa`` (to compute inharmonicity), or
``inharmonicity_b`` directly.
"""

import tomllib
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

DEFAULT_STRING_SET = "daddario_ej46"
LBF_TO_N = 4.4482216152605

_BUNDLED = files("spectrum_modeler") / "data" / "string_sets"


@dataclass(frozen=True)
class StringSpec:
    note: str
    tension: float  # N
    item: str | None = None  # maker's part number
    diameter: float | None = None  # m
    youngs_modulus: float | None = None  # Pa
    inharmonicity_b: float | None = None  # B, if known directly


@dataclass(frozen=True, eq=False)
class StringSet:
    key: str
    name: str
    source: str | None
    strings: dict[str, StringSpec]

    def __getitem__(self, note: str) -> StringSpec:
        try:
            return self.strings[note]
        except KeyError:
            raise KeyError(f"String set {self.key!r} has no {note} string") from None


def available_string_sets() -> list[str]:
    """Keys of the bundled string sets."""
    return sorted(
        p.name.removesuffix(".toml") for p in _BUNDLED.iterdir() if p.name.endswith(".toml")
    )


def load_string_set(name_or_path: str | Path = DEFAULT_STRING_SET) -> StringSet:
    """Load a bundled set by key (``"daddario_ej46"``) or any ``.toml`` file by path."""
    path = Path(name_or_path)
    if path.suffix == ".toml":
        return _parse(tomllib.loads(path.read_text()), key=path.stem)

    resource = _BUNDLED / f"{name_or_path}.toml"
    if not resource.is_file():
        raise ValueError(
            f"Unknown string set {name_or_path!r}; bundled sets: {available_string_sets()}"
        )
    return _parse(tomllib.loads(resource.read_text()), key=str(name_or_path))


def _parse(data: dict[str, Any], key: str) -> StringSet:
    strings = {
        note: _parse_string(note, entry, key) for note, entry in data.get("strings", {}).items()
    }
    if not strings:
        raise ValueError(f"String set {key!r} defines no strings")
    return StringSet(
        key=key, name=data.get("name", key), source=data.get("source"), strings=strings
    )


def _parse_string(note: str, entry: dict[str, Any], key: str) -> StringSpec:
    has_lbf, has_n = "tension_lbf" in entry, "tension_n" in entry
    if has_lbf == has_n:
        raise ValueError(f"{key} {note}: give exactly one of tension_lbf or tension_n")
    tension = entry["tension_lbf"] * LBF_TO_N if has_lbf else entry["tension_n"]
    if tension <= 0:
        raise ValueError(f"{key} {note}: tension must be positive")

    diameter_mm = entry.get("diameter_mm")
    modulus_gpa = entry.get("youngs_modulus_gpa")
    return StringSpec(
        note=note,
        tension=float(tension),
        item=entry.get("item"),
        diameter=None if diameter_mm is None else diameter_mm / 1000,
        youngs_modulus=None if modulus_gpa is None else modulus_gpa * 1e9,
        inharmonicity_b=entry.get("inharmonicity_b"),
    )
