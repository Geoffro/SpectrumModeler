import matplotlib.pyplot as plt
import pytest

from spectrum_modeler.modal import coefficients
from spectrum_modeler.presets import open_string, pluck
from spectrum_modeler.viz.shape import plot_mode_decomposition
from spectrum_modeler.viz.spectrum import plot_comparison, plot_spectrum
from spectrum_modeler.viz.style import ordinal_blues


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_comparison_figure_has_shape_and_spectrum_panels():
    e2 = open_string("E2")
    fig = plot_comparison(coefficients(e2, pluck("ponticello")), coefficients(e2, pluck("tasto")))
    assert len(fig.axes) == 4


def test_spectrum_bars_match_harmonic_count():
    ax = plot_spectrum(coefficients(open_string("G3"), pluck("twelfth_fret")))
    assert len(ax.patches) == 40


def test_mode_decomposition_renders():
    fig = plot_mode_decomposition(open_string("E4"), pluck("normal"))
    assert len(fig.axes) == 2


def test_ordinal_ramp_limits():
    assert len(ordinal_blues(4)) == 4
    with pytest.raises(ValueError):
        ordinal_blues(20)


def test_measurement_plot_renders_with_and_without_model():
    import numpy as np

    from spectrum_modeler.measure import measure, model_for
    from spectrum_modeler.synthesis import render
    from spectrum_modeler.viz.measured import plot_measurement

    sr = 48_000
    audio = np.concatenate(
        [np.zeros(4800), 0.3 * render(open_string("G3"), pluck("normal"), duration=1.0)]
    )
    audio += np.random.default_rng(0).normal(0, 1e-4, audio.size)
    with_note = measure(audio, sr, note="G3", pluck_mm=130.0, finger="ideal")
    bare = measure(audio, sr)
    for m in (with_note, bare):
        fig = plot_measurement(audio, sr, round(m.onset * sr), m, model_for(m))
        assert len(fig.axes) == 4  # three panels + colorbar


def test_calibration_report_renders(calibration_measurements):
    from spectrum_modeler.calibrate import calibrate
    from spectrum_modeler.synthetic import truth_response_db
    from spectrum_modeler.viz.calibration import plot_calibration_report

    _, measurements = calibration_measurements
    fig = plot_calibration_report(calibrate(measurements, name="x"), truth_response_db)
    assert len(fig.axes) == 3


def test_brightness_vs_position_has_one_line_per_finger():
    import numpy as np

    from spectrum_modeler.viz.spectrum import plot_brightness_vs_position

    ax = plot_brightness_vs_position(open_string("E2"), np.arange(20, 330, 10))
    centroids = {line.get_label(): line.get_ydata() for line in ax.lines}
    assert set(centroids) == {"ideal", "nail", "flesh"}
    assert np.all(centroids["flesh"] <= centroids["ideal"] + 1e-9)  # the fingertip softens


def test_themes_switch_and_restore():
    from spectrum_modeler.viz import style

    assert style.current_theme().name == "light"
    style.set_theme("dark")
    try:
        assert style.C.SURFACE == style.DARK.SURFACE
        with pytest.raises(RuntimeError), style.using_theme("light"):
            assert style.C.SURFACE == style.LIGHT.SURFACE
            raise RuntimeError
        assert style.current_theme().name == "dark"
    finally:
        style.set_theme("light")


def test_figures_outside_pyplot_are_laid_out():
    """The app's figures (and its exports) are not pyplot's; layout must still run."""
    from spectrum_modeler.fdtd import simulate
    from spectrum_modeler.viz import style
    from spectrum_modeler.viz.animate import string_frames

    string, p = open_string("E4"), pluck("ponticello")
    result = simulate(string, p, 0.01, snapshot_every=8, snapshot_until=1 / string.f0)
    style.use_pyplot(False)
    try:
        frames = string_frames(result, slowdown=20, fps=10)
    finally:
        style.use_pyplot(True)
    assert not plt.get_fignums()  # no pyplot window opened
    assert frames.fig.axes[0].get_position().x0 != pytest.approx(0.125)  # not the default
