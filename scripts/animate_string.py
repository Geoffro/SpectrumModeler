"""Simulate the plucked string in time and animate it, for one or more pluck positions.

For each position, writes to --out:
    <note>_<position>.mp4 (or .gif)     slow-motion animation with the bridge force
    <note>_<position>_filmstrip.png     snapshots through one period
    <note>_<position>_check.png         simulated harmonics against the modal model

Examples:
    python scripts/animate_string.py                                  # E2, ponticello and tasto
    python scripts/animate_string.py --string G3 --positions 40 twelfth_fret --format gif
    python scripts/animate_string.py --damping --periods 20 --slowdown 50
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from spectrum_modeler.modal import coefficients
from spectrum_modeler.params import ModelOptions
from spectrum_modeler.presets import (
    FINGER_WIDTHS_MM,
    OPEN_STRING_NOTES,
    PLUCK_PRESETS,
    open_string,
    pluck,
)
from spectrum_modeler.viz.animate import (
    animate_string,
    plot_filmstrip,
    plot_modal_check,
    save_animation,
    simulate_for_animation,
)


def position(value: str) -> str | float:
    try:
        return float(value)
    except ValueError:
        if value not in PLUCK_PRESETS:
            raise argparse.ArgumentTypeError(
                f"{value!r}: use mm or one of {PLUCK_PRESETS}"
            ) from None
        return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--string", default="E2", choices=OPEN_STRING_NOTES)
    parser.add_argument("--positions", nargs="+", type=position, default=["ponticello", "tasto"])
    parser.add_argument("--finger", default="ideal", choices=list(FINGER_WIDTHS_MM))
    parser.add_argument("--periods", type=float, default=2, help="how many periods to animate")
    parser.add_argument("--slowdown", type=float, default=250, help="times slower than real time")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--damping", action="store_true", help="include the string's losses")
    parser.add_argument("--format", choices=["mp4", "gif"], default="mp4")
    parser.add_argument("--out", type=Path, default=Path("output/animation"))
    args = parser.parse_args()

    string = open_string(args.string)
    args.out.mkdir(parents=True, exist_ok=True)
    for pos in args.positions:
        p = pluck(pos, finger=args.finger)
        stem = f"{string.note}_{pos if isinstance(pos, str) else f'{pos:g}mm'}"
        result = simulate_for_animation(
            string, p, periods=args.periods, slowdown=args.slowdown, fps=args.fps,
            damping=args.damping,
        )  # fmt: skip

        modal = coefficients(string, p, ModelOptions(n_harmonics=20, damping=False))
        if not args.damping:
            fig = plot_modal_check(result, modal)
            fig.savefig(args.out / f"{stem}_check.png")
            plt.close(fig)
        fig = plot_filmstrip(result)
        fig.savefig(args.out / f"{stem}_filmstrip.png")
        plt.close(fig)
        path = save_animation(
            animate_string(result, slowdown=args.slowdown, fps=args.fps),
            args.out / f"{stem}.{args.format}",
            fps=args.fps,
        )
        print(f"wrote {path} ({result.x.size - 1} grid intervals, courant {result.courant:.3f})")
    print(f"filmstrips and checks in {args.out}")


if __name__ == "__main__":
    main()
