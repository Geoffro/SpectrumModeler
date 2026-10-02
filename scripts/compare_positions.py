"""Compare two pluck positions on the open strings: spectra, spectrograms, and audio.

For each string, writes to --out:
    <note>_<a>_vs_<b>.png               string shapes and harmonic spectra
    <note>_<a>_vs_<b>_spectrogram.png   how the harmonics decay
    <note>_<a>.wav, <note>_<b>.wav      each pluck, matched in loudness
    <note>_<a>_then_<b>.wav             both, one after the other, for A/B listening

Examples:
    python scripts/compare_positions.py                      # all strings, ponticello vs tasto
    python scripts/compare_positions.py --string E2 --a 40 --b twelfth_fret
    python scripts/compare_positions.py --finger nail --body --no-damping
    python scripts/compare_positions.py --damping flat       # every partial decays together
    python scripts/compare_positions.py --sigma2 0.5e-6      # less high-frequency loss
    python scripts/compare_positions.py --detail             # also plot mode decompositions
    python scripts/compare_positions.py --profile my_guitar  # calibrated to your guitar
"""

import argparse
import dataclasses
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from spectrum_modeler.analysis import spectrogram
from spectrum_modeler.audio import match_loudness, sequence, write_wav
from spectrum_modeler.modal import coefficients
from spectrum_modeler.params import DAMPING_MODES, OBSERVABLES, ModelOptions
from spectrum_modeler.presets import (
    DEFAULT_SCALE_LENGTH,
    FINGER_WIDTHS_MM,
    OPEN_STRING_NOTES,
    PLUCK_PRESETS,
    open_strings,
    pluck,
)
from spectrum_modeler.string_sets import DEFAULT_STRING_SET, load_string_set
from spectrum_modeler.synthesis import DEFAULT_SAMPLE_RATE, render_spectrum
from spectrum_modeler.viz.shape import plot_mode_decomposition
from spectrum_modeler.viz.spectrogram import plot_spectrogram_pair
from spectrum_modeler.viz.spectrum import describe, plot_comparison


def position(value: str) -> str | float:
    """A preset name or a distance from the saddle in mm."""
    try:
        return float(value)
    except ValueError:
        if value not in PLUCK_PRESETS:
            raise argparse.ArgumentTypeError(
                f"{value!r} is neither a distance in mm nor one of {PLUCK_PRESETS}"
            ) from None
        return value


def slug(pos: str | float) -> str:
    return pos if isinstance(pos, str) else f"{pos:g}mm"


def save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--string", default="all", choices=["all", *OPEN_STRING_NOTES])
    parser.add_argument("--a", type=position, default="ponticello", help="preset or mm")
    parser.add_argument("--b", type=position, default="tasto", help="preset or mm")
    parser.add_argument("--finger", default="flesh", choices=list(FINGER_WIDTHS_MM))
    parser.add_argument("--observable", default="bridge_force", choices=OBSERVABLES)
    parser.add_argument("--harmonics", type=int, default=40)
    parser.add_argument("--string-set", default=DEFAULT_STRING_SET, help="key or .toml path")
    parser.add_argument(
        "--scale-length", type=float, default=DEFAULT_SCALE_LENGTH * 1000, help="mm"
    )
    layers = parser.add_argument_group("realism layers")
    layers.add_argument(
        "--damping",
        default="full",
        choices=DAMPING_MODES,
        help="full: sigma0 + sigma2 f^2; flat: sigma0 only; off: no decay",
    )
    layers.add_argument(
        "--no-damping", dest="damping", action="store_const", const="off", help="--damping off"
    )
    layers.add_argument("--sigma0", type=float, help="1/s, replaces the string's (or profile's)")
    layers.add_argument(
        "--sigma2", type=float, help="1/(s Hz^2), replaces the string's (or profile's)"
    )
    layers.add_argument("--no-finger-width", action="store_true")
    layers.add_argument(
        "--inharmonicity",
        type=float,
        metavar="B",
        help="switch on inharmonicity with this B for every string (overrides the string set)",
    )
    layers.add_argument("--body", action="store_true", help="illustrative body resonances")
    layers.add_argument(
        "--profile", help="guitar profile (name or .json): calibrated values for every layer"
    )
    parser.add_argument("--duration", type=float, default=4.0, help="seconds of audio")
    parser.add_argument("--no-audio", action="store_true", help="skip WAVs and spectrograms")
    parser.add_argument("--detail", action="store_true", help="also plot mode decompositions")
    parser.add_argument("--out", type=Path, default=Path("output"))
    args = parser.parse_args()

    length = args.scale_length / 1000
    strings = open_strings(string_set=load_string_set(args.string_set), length=length)
    if args.string != "all":
        strings = [s for s in strings if s.note == args.string]
    if args.inharmonicity is not None:
        strings = [dataclasses.replace(s, inharmonicity_b=args.inharmonicity) for s in strings]
    opts = ModelOptions(
        n_harmonics=args.harmonics,
        observable=args.observable,
        damping=args.damping,
        finger_width=not args.no_finger_width,
        inharmonicity=True if args.inharmonicity is not None else None,
        body=True if args.body else None,
        profile=args.profile,
        sigma0=args.sigma0,
        sigma2=args.sigma2,
    )
    pluck_a = pluck(args.a, length=length, finger=args.finger)
    pluck_b = pluck(args.b, length=length, finger=args.finger)
    args.out.mkdir(parents=True, exist_ok=True)

    for string in strings:
        spec_a, spec_b = coefficients(string, pluck_a, opts), coefficients(string, pluck_b, opts)
        stem = f"{string.note}_{slug(args.a)}_vs_{slug(args.b)}"
        save(plot_comparison(spec_a, spec_b), args.out / f"{stem}.png")

        if not args.no_audio:
            sr = DEFAULT_SAMPLE_RATE
            audio_a, audio_b = match_loudness(
                render_spectrum(spec_a, args.duration, sr, normalize=False),
                render_spectrum(spec_b, args.duration, sr, normalize=False),
            )
            for pos, audio in ((args.a, audio_a), (args.b, audio_b)):
                path = args.out / f"{string.note}_{slug(pos)}.wav"
                write_wav(path, audio, sr)
                print(f"wrote {path}")
            path = args.out / f"{string.note}_{slug(args.a)}_then_{slug(args.b)}.wav"
            write_wav(path, sequence([audio_a, audio_b], sr), sr)
            print(f"wrote {path}")

            fig = plot_spectrogram_pair(
                spectrogram(audio_a, sr),
                spectrogram(audio_b, sr),
                labels=(describe(spec_a), describe(spec_b)),
                suptitle=f"{string.note}: spectrogram, matched loudness",
            )
            save(fig, args.out / f"{stem}_spectrogram.png")

        if args.detail:
            for pos, p in ((args.a, pluck_a), (args.b, pluck_b)):
                save(
                    plot_mode_decomposition(string, p),
                    args.out / f"{string.note}_{slug(pos)}_modes.png",
                )


if __name__ == "__main__":
    main()
