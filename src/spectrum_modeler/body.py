"""Body response: how the guitar colors the bridge force (design §2.3d).

Until a calibrated response curve exists, a small bank of peaking filters stands
in for the air resonance and the first top-plate modes. The values are
illustrative, not measured. The response is applied as a gain on each partial.
"""

from dataclasses import dataclass

import numpy as np
from scipy import signal

DESIGN_SAMPLE_RATE = 48_000


@dataclass(frozen=True)
class Resonance:
    freq: float  # Hz
    q: float
    gain_db: float


ILLUSTRATIVE_RESONANCES = (
    Resonance(freq=100.0, q=5.0, gain_db=12.0),  # air (Helmholtz) resonance
    Resonance(freq=200.0, q=6.0, gain_db=9.0),  # main top-plate mode
    Resonance(freq=400.0, q=4.0, gain_db=6.0),  # higher top-plate mode
)


def peaking_sos(
    resonances: tuple[Resonance, ...], sample_rate: float = DESIGN_SAMPLE_RATE
) -> np.ndarray:
    """Peaking-EQ biquads (RBJ cookbook), one second-order section per resonance."""
    sections = []
    for r in resonances:
        a_gain = 10 ** (r.gain_db / 40)
        w0 = 2 * np.pi * r.freq / sample_rate
        alpha = np.sin(w0) / (2 * r.q)
        b = [1 + alpha * a_gain, -2 * np.cos(w0), 1 - alpha * a_gain]
        a = [1 + alpha / a_gain, -2 * np.cos(w0), 1 - alpha / a_gain]
        sections.append([*(np.array(b) / a[0]), 1.0, a[1] / a[0], a[2] / a[0]])
    return np.array(sections)


def response_db(
    freqs: np.ndarray, resonances: tuple[Resonance, ...] = ILLUSTRATIVE_RESONANCES
) -> np.ndarray:
    """Gain in dB of the resonance bank at each frequency in ``freqs`` (Hz)."""
    freqs = np.asarray(freqs, dtype=float)
    _, h = signal.sosfreqz(peaking_sos(resonances), worN=freqs, fs=DESIGN_SAMPLE_RATE)
    return 20 * np.log10(np.abs(h))
