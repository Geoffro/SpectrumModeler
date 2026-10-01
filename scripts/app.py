"""Open the SpectrumModeler app: explorer, comparison, animation, and tuner in one window.

Tabs:
    Explorer   drag the pluck point, and watch and hear the overtones change
    Compare    two pluck positions side by side; export plots and WAVs
    Animate    the simulated string in slow motion; export MP4 or GIF
    Tuner      adjust a guitar profile against measured takes

Open a recording session and a guitar profile from the bar at the top; the app
remembers them for next time.

Example:
    python scripts/app.py
"""

import matplotlib

matplotlib.use("TkAgg")
import tkinter as tk

from spectrum_modeler.viz.app import App


def main() -> None:
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
