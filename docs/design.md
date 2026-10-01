# SpectrumModeler — Design Document

**Language:** Python 3.12 (conda environment)
**Roadmap and progress tracking:** [roadmap.md](roadmap.md)
**Recording process and known concerns:** [recording-guide.md](recording-guide.md)

---

## 1. Overview

SpectrumModeler is a Python program that models a plucked classical guitar string. It shows how the **pluck position** changes the string's **harmonic content** and so its **timbre**. The main comparison is a pluck near the bridge (*sul ponticello*: bright and nasal) against a pluck over or beyond the soundhole (*sul tasto*: warm and round).

The program has four jobs:

1. **Compute** the theoretical amplitude of every overtone for any pluck position.
2. **Measure** real recordings of the guitar (made on an iPhone) and extract their overtone content.
3. **Calibrate** the theoretical model against those measurements, so that it describes this particular guitar.
4. **Visualize and synthesize.** Plot the overtones and play the model as audio, next to the real recordings, so the difference can be both seen and heard.

### 1.1 Goals

- Physically correct results for the ideal string, checked against analytic results by tests.
- Realism added in layers (damping, finger width, stiffness, body response). Each layer can be switched on or off so its effect can be heard in isolation.
- A repeatable way to record, import, and measure real plucks, with the results stored alongside the recordings.
- A calibration step, automatic fitting followed by optional manual adjustment, that saves a reusable **guitar profile**.
- Side-by-side comparison of model and recording, and of two pluck positions, with plots and audio.
- Built like a real project: version control, a package layout, tests, linting, and documentation.

### 1.2 Version 1 scope

- **The six open strings only**, in standard tuning (E2 A2 D3 G3 B3 E4).
- Any pluck position along the open string.
- One guitar and one calibration profile at a time.

### 1.3 Non-goals (for version 1)

- Fretted notes. The data model keeps the pluck position in millimetres from the saddle, so fretted notes can be added later without redesign (see §10).
- Real-time audio synthesis with a low-latency GUI.
- A full physical model of the guitar body (for example, finite-element models of the top plate). The body is represented by a response curve measured from recordings (§4.5).
- Coupling between strings, or modeling string polarization in two planes.
- A web or browser version.

---

## 2. Physics Background

### 2.1 Ideal string: modal solution

The string has length $L$, tension $T$, and linear density $\mu$. Waves travel along it at speed $c = \sqrt{T/\mu}$, so the fundamental frequency is $f_0 = c / 2L$.

The finger pulls the string to a height $h$ at distance $x_p = \beta L$ from the bridge, where $0 < \beta < 1$, and releases it from rest. The starting shape is a triangle. Solving the wave equation with fixed ends gives:

$$
y(x,t) = \sum_{n=1}^{\infty} a_n \sin\!\left(\frac{n\pi x}{L}\right)\cos(2\pi n f_0 t)
$$

$$
a_n = \frac{2h}{n^2\pi^2\,\beta(1-\beta)}\,\sin(n\pi\beta)
$$

Source: E. M. Stein and R. Shakarchi, *Fourier Analysis: An Introduction* (Princeton, 2003), from "Example: the plucked string"

**Main consequences:**

| Effect | Cause | What you hear |
|---|---|---|
| Overall $1/n^2$ roll-off | The triangle has a corner (its slope jumps) | Mellow baseline |
| Missing harmonics wherever $n\beta$ is a whole number | The pluck point sits on a node of mode $n$ | Hollow, "filtered" character |
| Small $\beta$ (near the bridge) | $\sin(n\pi\beta)$ keeps growing through many harmonics before its first zero at $n = 1/\beta$ | Bright, thin, nasal |
| $\beta = 1/2$ (12th fret) | All even harmonics are missing | Round, clarinet-like |

### 2.2 What the listener hears: bridge force

The soundboard is driven by the force the string puts on the bridge. That force is proportional to the string's slope at $x = 0$:

$$
F(t) = T\,\frac{\partial y}{\partial x}\Big|_{x=0} = \sum_n \underbrace{T\,a_n\,\frac{n\pi}{L}}_{F_n}\cos(2\pi n f_0 t),
\qquad F_n \propto \frac{\sin(n\pi\beta)}{n}
$$

So the bridge force falls off as **1/n**, not $1/n^2$. It is the more relevant spectrum for the perceived sound. The model offers three **observables**:

- `displacement` — the $a_n$ values (the string's shape)
- `velocity` — $2\pi f_n a_n$, the modal velocity
- `bridge_force` — $F_n$ above (the default for audio and for calibration)

### 2.3 Realism layers

Each layer is a multiplier on the mode amplitudes, a change to the mode frequencies, or a filter on the output. Each can be switched on or off separately. The parameters of every layer can be set by hand or fitted from recordings (§4.5).

**(a) Frequency-dependent damping.** Mode $n$ decays as $e^{-\sigma_n t}$, with

$$
\sigma_n = \sigma_0 + \sigma_2\,(n f_0)^2
$$

$\sigma_0$ accounts for air and support losses, and $\sigma_2$ for internal friction in the string. High harmonics die out first, so the note gets darker over time. Where a calibration profile exists, the measured decay rate for each harmonic can be used directly.

**(b) Finite plucking width.** A fingertip or nail of width $w$ rounds off the triangle's corner. As an approximation, smoothing the initial shape with a box of width $w$ multiplies each amplitude by

$$
S_n = \operatorname{sinc}\!\left(\frac{n\pi w}{2L}\right), \quad \operatorname{sinc}(u) = \frac{\sin u}{u}
$$

This gives presets such as `nail` (w ≈ 1–2 mm) and `flesh` (w ≈ 8–15 mm), which calibration then refines.

**(c) Stiffness (inharmonicity).** Bending stiffness raises the upper partials:

$$
f_n = n f_0 \sqrt{1 + B n^2}, \qquad B = \frac{\pi^3 E d^4}{64\,T L^2}
$$

For nylon, $B$ is small, but it is measurable on the bass strings (which have a wound nylon core). Calibration estimates $B$ directly from the measured partial frequencies. Without a profile, this layer is off by default.

**(d) Body and recording response.** A response curve $H(f)$ in dB represents everything between the bridge force and the recorded sound: the soundboard and air resonances, the room, and the iPhone microphone.

- **Calibrated (preferred):** $H(f)$ is measured from recordings (§4.5).
- **Uncalibrated fallback:** a small bank of illustrative resonant (biquad) filters: the air resonance near 90–110 Hz and the top-plate modes near 180–220 Hz and about 400 Hz.

Either way, $H(f)$ is applied as a gain on each partial's amplitude, not as a filter on the audio. The spectrum plots and the synthesized audio therefore always agree. This leaves out the body's own brief ringing at the attack, which is acceptable here.

### 2.4 Time-domain simulation

`fdtd.py` solves the string's equation of motion directly on a grid, step by step:

$$
\frac{\partial^2 y}{\partial t^2} = c^2 \frac{\partial^2 y}{\partial x^2} - \kappa^2 \frac{\partial^4 y}{\partial x^4} - 2\sigma_0 \frac{\partial y}{\partial t} + 2\sigma_1 \frac{\partial^3 y}{\partial t\,\partial x^2}
$$

with the ends fixed (and simply supported, for the stiffness term). The coefficients match the modal model: $c = 2Lf_0$, $\sigma_1 = \sigma_2 c^2/4\pi^2$ (giving a decay rate of $\sigma_0 + \sigma_2 f^2$), and $\kappa^2 = B c^2 L^2/\pi^2$ (giving partials at $n f_0\sqrt{1+Bn^2}$).

- **Scheme:** explicit, centred differences in time and space, with the frequency-dependent loss as a backward time difference. The pluck is released from rest; a finger's width is applied through the grid's sine modes, as in the modal model.
- **Stability:** the grid spacing must satisfy $h \ge \sqrt{(a + \sqrt{a^2 + 16\kappa^2 k^2})/2}$ with $a = c^2k^2 + 4\sigma_1 k$ (the Courant condition, generalized). By default the finest stable grid is used, and a requested grid that is too fine is refused. The ideal string runs exactly at $c\,k/h = 1$, where the scheme is exact: every mode oscillates at exactly $n f_0$.
- **Bridge force:** tension times the slope at $x = 0$, taken spectrally from the grid's sine modes, with the upper half of the modes tapered so the force steps don't ring. A one-sided difference would overstate the upper harmonics on a coarse grid.
- **Energy:** the scheme's discrete energy is tracked on request. It is constant to round-off (about $10^{-15}$) without loss, and never increases with loss.

**Validation:** the simulated bridge force matches the modal $F_n$ within 0.5 dB for the first 20 harmonics (those within 40 dB of the strongest), on all six strings at four positions, ideal and flesh.

**Output** (`scripts/animate_string.py`): a slow-motion animation (MP4 or GIF) of the string with the bridge force traced below; a filmstrip of one period, showing the two kinks travelling out from the pluck point and reflecting (Helmholtz motion); and a check plot of simulated against modal harmonics.

---

## 3. Reference Parameters

### 3.1 Instrument defaults

| Parameter | Default | Notes |
|---|---|---|
| Scale length $L$ | 650 mm | Standard classical guitar; replace with your guitar's measured length |
| Pluck amplitude $h$ | 2 mm | Only sets the overall level |
| Sample rate | 48 000 Hz | Matches iPhone recordings |
| Harmonics computed | 40 (configurable) | Enough for the bright end of the range |

### 3.2 Open strings (version 1 scope)

Default string set: **D'Addario Pro-Arté Nylon, Hard Tension (EJ46)**. Tensions are from the product page; newtons are converted using 1 lbf = 4.448 N.

| String | Note | $f_0$ (Hz) | Item | Tension (lbf) | Tension (N) | $\mu$ (g/m, derived) |
|---|---|---|---|---|---|---|
| 6 | E2 | 82.41 | J4606 | 15.2 | 67.6 | 5.89 |
| 5 | A2 | 110.00 | J4605 | 17.0 | 75.6 | 3.70 |
| 4 | D3 | 146.83 | J4604 | 16.2 | 72.1 | 1.98 |
| 3 | G3 | 196.00 | J4603 | 12.3 | 54.7 | 0.84 |
| 2 | B3 | 246.94 | J4602 | 12.4 | 55.2 | 0.54 |
| 1 | E4 | 329.63 | J4601 | 16.8 | 74.7 | 0.41 |

The frequencies are nominal. When a recording is analyzed, its actual $f_0$ is measured (§4.4), so slight detuning does not affect the analysis.

**Linear density from tension.** Once tension, $f_0$, and $L$ are known, the linear density follows directly: $\mu = T / (2 L f_0)^2$. The last column was computed this way for $L$ = 650 mm and is recomputed from your measured scale length. $\mu$ is needed only by the time-domain simulation. The modal model needs only $f_0$.

**Stiffness.** Computing $B$ from first principles also needs each string's diameter and Young's modulus, which the product page does not give. These fields are optional. When they are missing, $B$ is taken from calibration (§4.4), which is more accurate anyway for wound bass strings.

#### String sets are configurable

String sets are TOML files in `src/spectrum_modeler/data/string_sets/`. Add a file to describe other strings, then select it with `--string-set <name>` on the scripts, or `open_string(note, string_set=...)`. A guitar profile records which set it was calibrated with.

```toml
# daddario_ej46.toml
name = "D'Addario Pro-Arté Nylon, Hard Tension (EJ46)"
source = "D'Addario product page"

[strings.E4]
item = "J4601"
tension_lbf = 16.8
# diameter_mm = ...         # optional: needed only for B from first principles
# youngs_modulus_gpa = ...  # optional

[strings.B3]
item = "J4602"
tension_lbf = 12.4
# ... G3, D3, A2, E2
```

Tension may be given as `tension_lbf` or `tension_n`; it is converted to newtons on load. Tests check that the bundled set loads and that the derived $\mu$ reproduces $f_0$.

### 3.3 Pluck-position presets

Positions are stored in **millimetres from the saddle**; β is derived from them. Measure your own instrument and adjust.

| Preset | Distance | $\beta$ | Character |
|---|---|---|---|
| `ponticello` | ~40 mm | ~0.06 | Bright, metallic |
| `normal` | ~130 mm | ~0.20 | Balanced (usual playing spot, near the soundhole) |
| `tasto` | ~250 mm | ~0.38 | Warm, soft |
| `twelfth_fret` | 325 mm | 0.50 | No even harmonics |

---

## 4. Recordings and Calibration

The idea: the model accounts for what the **pluck** does. A recording contains the pluck's effect *plus* everything the guitar body, the room, and the microphone add. With the same string recorded at several pluck positions, the pluck's part (which changes with position) can be separated from the rest (which does not). What remains is the guitar's own response, $H(f)$.

### 4.1 Recording protocol

The step-by-step process, the session checklist, the known concerns about recording with an iPhone, and the criteria for judging whether iPhone recordings are good enough are all in **[recording-guide.md](recording-guide.md)**.

The first pass uses the iPhone's Voice Memos app. The pipeline must not depend on it: any file that ffmpeg can read (`.m4a`, `.wav`, `.flac`, `.aiff`, `.mp3`) is accepted, and each clip records which recorder made it. If the iPhone proves inadequate, a better recording setup can be substituted, and only the recordings and the profile need to be redone.

**All code is built before any recording is made.** Two things make that possible:

- **Synthetic sessions.** `synthetic.py` renders a complete fake session: clips with several plucks each, a room-tone clip, and a manifest. It uses known values of β, $w$, $\sigma_0$, $\sigma_2$, $B$, $H(f)$, per-take gain, and noise, and can inject impairments (clipping, a lossy codec, a missing manifest row, sympathetic peaks). The whole pipeline (import → measure → calibrate → tuner → explorer) is tested against it, and the fitted values are checked against the known ones.
- **Ad-hoc mode.** `scripts/analyze.py <file>` measures and plots any single audio file without a manifest. The note can be given with `--note` or `--f0`; otherwise $f_0$ is detected automatically. This is useful for trying the measurement code on arbitrary MP3s, a guitar sample, or a quick phone recording.

A **session directory** has the same layout wherever it lives: `manifest.csv`, `raw/`, `processed/`, `measurements/`. Real recordings use `recordings/`; synthetic sessions default to `output/synthetic_session/`. Every script takes `--session <dir>`.

Summary of the version 1 recording plan:

| Session | Strings | Positions (mm from saddle) | Finger | Plucks per clip | Clips |
|---|---|---|---|---|---|
| Pilot | E2 and G3 | 40, 80, 130, 200, 325 | flesh | 3 | 10 |
| Full set | all 6 | same | flesh | 3 | 30 |
| Nail comparison (optional) | all 6 | same | nail | 3 | 30 |

### 4.2 File naming and manifest

Each clip covers one string, one position, and one finger type, and contains several plucks. The import step splits it into separate takes. Clips are renamed after transfer and placed in `recordings/raw/`:

```
<note>_<pos>mm_<finger>.<ext>
e.g.  E2_040mm_flesh.m4a
```

Special clips: `ROOMTONE_<session>.m4a` (silence, for measuring the noise floor).

Each clip also has a row in `recordings/manifest.csv`, which records details that cannot be put in a filename:

```csv
file,string,note,pluck_mm,finger,stroke,plucks,session,recorder,mic_distance_cm,notes
E2_040mm_flesh.m4a,6,E2,40,flesh,free,3,2026-10-04,iphone_voice_memos,40,
```

The import script checks that every file has a manifest row, that every row has a file, and that the number of detected plucks matches `plucks`. Any mismatch is flagged.

### 4.3 Import pipeline

`scripts/import_recordings.py --session <dir>` (module `recording.py`):

1. **Check the manifest.** Error for files in `raw/` with no manifest row, and for rows with no file.
2. **Inspect.** Use `ffprobe` to report the codec, sample rate, and channel count. Warn on lossy codecs (AAC, MP3). For iPhone clips, AAC means Voice Memos was not set to Lossless; lossless iPhone recordings use ALAC. Lossy files are still processed, for testing.
3. **Decode.** Convert each file with `ffmpeg` to 48 kHz, mono, 32-bit float, cached as `processed/<clip>.wav` until the raw file changes. Stereo is mixed down to mono.
4. **Find plucks.** A pluck starts where the 5 ms level jumps 20 dB above both the quietest level of the preceding 50 ms and the clip's noise floor (its 1st-percentile level). The onset is refined to the first sample 20 dB above the preceding quiet. Detection re-arms only once the level has fallen back within 10 dB of the floor, which is why the recording guide asks for each pluck to be damped.
5. **Split into takes.** Each take runs from 20 ms before its onset to 20 ms before the next, and is written as `processed/<clip>_t<n>.wav`.
6. **Check quality.** Warn on clipping (runs of 3 or more samples at full scale), a peak below −30 dBFS, a noise floor more than 3 dB above the room tone, or less than 40 dB between peak and floor. Warn if the number of plucks found differs from the manifest.
7. **Report.** Save everything, including every issue, to `processed/import_report.json`. The script exits with status 1 if there are errors.

### 4.4 Measurement (per take)

`scripts/analyze.py <session>` measures every take; `scripts/analyze.py <file>` measures the first pluck in any audio file (module `measure.py`). Times are relative to the onset.

1. **Analysis window.** 30–180 ms after the onset: late enough to skip the attack, early enough that fast-decaying high partials are still present. It uses a Blackman-Harris window (sidelobes about 92 dB down), zero-padded 8×, so a strong partial's sidelobes don't make a missing neighbour look present.
2. **Fundamental.** A harmonic-sum search scores each candidate $f_0$ by the height of its first 8 harmonics above the median level. The search is limited to ±50 cents of the nominal $f_0$ when the note is known, and to 60–1000 Hz otherwise. Scoring many harmonics keeps a pluck with missing harmonics (at a node) from being mistaken for another pitch.
3. **Harmonics.** Each harmonic is found in turn near $n f_0 \sqrt{1+Bn^2}$, with $f_0$ and $B$ refitted as reliable peaks accumulate, so the search follows the stretched partials. A harmonic counts as **reliable** only if all three hold:
   - it is a local maximum inside the search range (not a neighbour's skirt);
   - it is at least 15 dB above the local floor (the median level between its main lobe and half a harmonic spacing away);
   - it lies within one frequency bin (6.7 Hz) of the prediction. Noise peaks land anywhere in the range.
4. **Stiffness.** Fit $f_0$ and $B$ from the reliable peaks: $(f_n/n)^2 = f_0^2 + f_0^2 B\,n^2$ is linear in $n^2$. $B$ is reported when at least 5 harmonics are reliable.
5. **Decay.** An STFT (80 ms Hann, 10 ms hop) follows each reliable harmonic. A straight line fitted to its dB level, over the first unbroken run of frames at least 15 dB above the local floor, gives $\sigma_n$. The fit stops early if the note is damped, which shows as a broadband drop of more than 7.5 dB within 50 ms. $\sigma_0$ and $\sigma_2$ are then fitted with errors relative to the fitted rate, so the slow low partials, which set $\sigma_0$, count as much as the fast high ones; rates more than 4 robust deviations off the fit are dropped.
6. **Onset levels.** The level in the window is extrapolated back to the onset using each harmonic's decay rate (or the fitted $\sigma_0 + \sigma_2 f^2$ where it was not measured). Unreliable harmonics, and those that decayed more than 20 dB by the window, get no onset level.
7. **Off-series peaks.** A 0.5 s Blackman-Harris spectrum is searched for peaks with at least 15 dB prominence, within 60 dB of the strongest peak and 20 dB above the floor, and more than $\max(0.1 f_0, 12\text{ Hz})$ from any harmonic. These flag sympathetic strings or distortion.
8. **Output.** A `Measurement` per take (frequency, window level, onset level, floor, decay rate, and a reliability flag per harmonic, plus $f_0$, $B$, $\sigma_0$, $\sigma_2$, off-series peaks, and metadata), saved to `measurements/<clip>.json`. There is one plot per clip (or per take, with `--plots all`) in `plots/`: the window spectrum with the harmonics marked, the onset levels against the uncalibrated model, and a spectrogram.

**Round-trip accuracy** (synthetic plucks with −80 dBFS noise, all six strings at 40, 130, and 325 mm): $f_0$ within 0.02 Hz, $B$ within 3%, $\sigma_0$ within 5%, $\sigma_2$ within 3%, and onset levels mostly within 2 dB of the model.

### 4.5 Calibration model

For string $s$, pluck position $p$, and harmonic $n$, the measured level in dB is modeled as:

$$
M_{s,p,n} = G_{s,p} + T_n(\beta_p, w) + H(f_{s,n}) + \varepsilon
$$

| Term | Meaning | How it is found |
|---|---|---|
| $T_n(\beta_p, w)$ | Theoretical bridge-force spectrum in dB, including finger width | From the physics (§2) |
| $G_{s,p}$ | Overall level of each take (pluck force and microphone gain are uncontrolled) | Fitted, one value per take |
| $H(f)$ | Body, room, and microphone response | Fitted: one smooth curve over frequency, shared by all strings |
| $w$ | Effective finger width | Fitted, one value per finger type |
| $\beta_p$ | Effective pluck position | Starts at the taped measurement; may be refined within ±5 mm as a check |

$T_n(\beta, w) = 20\log_{10}\left|\sin(n\pi\beta)\,S_n/n\right|$, with $S_n$ from §2.3b. Nodes are limited to 40 dB below the $S_n/n$ envelope. Real plucks never land exactly on a node, and an unlimited model node produces residuals of 70 dB or more, which swamp the fit.

$H$ depends on frequency but not on pluck position. Because each string is recorded at several positions, it can be separated from $T_n$: the dips at the nodes of the pluck point move with position, while body resonances stay at fixed frequencies.

**Fitting** (`calibrate.py`, `scripts/calibrate.py`):
- **Data:** every reliable harmonic with an onset level (§4.4) from every take with a note, pluck position, and finger.
- **$H(f)$:** a cubic B-spline in log frequency, 4 knots per octave, with a second-difference roughness penalty (weight 1.0) and its mean pinned to 0 dB. $G$ absorbs the constant. The spline spans the 1st–99th percentile of measured frequencies, because a spline end resting on a handful of points swings wildly. Beyond that range the profile holds $H$ constant.
- **Solver:** `scipy.optimize.least_squares` with a soft-L1 loss (3 dB scale) and a sparse Jacobian pattern. A 90-take fit takes under a second.
- **Positions:** fitted by alternating two steps. First, fit $G$, $H$, and $w$ with positions fixed. Then, for each take, grid-search its position within ±5 mm of the manifest in 0.1 mm steps, holding $H$ and $w$, with a Gaussian prior (σ = 2 mm) centred on the tape mark. Repeat twice. A global search per take is needed because a pluck near a node has several near-equal local minima. The prior keeps the choice near the mark unless the data clearly disagree.
- **Per string, directly from the measurements:** $f_0$ (median over takes), $B$ (median), $\sigma_0$ and $\sigma_2$ (robust weighted fit pooled over takes, §4.4), and the median measured $\sigma_n$ for each harmonic. Harmonics near body resonances often decay faster than the formula predicts, so the measured rates are used where available.

**Diagnostics** (`<session>/plots/calibration_<name>.png`):
- $H(f)$ with the data points behind it; for synthetic sessions, the true $H(f)$ as well;
- the RMS error per take, against the 3 dB target;
- the fitted position minus the manifest position per take. Shifts at the ±5 mm limit are flagged: check the tape mark.

**Round-trip accuracy** (synthetic session, all six strings at five positions, 90 takes, known $H$ with sharp resonances and a 90 Hz high-pass): $H$ within 0.5 dB (95th percentile) at the measured frequencies; $B$, $\sigma_0$, $\sigma_2$ within 1.5%; positions refined from 1.2 mm to 0.6–0.9 mm RMS; 99% of takes within 3 dB. A deliberately mis-taped clip (+4 mm) is pulled back to within about 1 mm. Finger width is only weakly determined, because it moves the highest harmonics by a dB or two: 9.8–11.5 mm against a true 11 mm, depending on the data. Treat it as an effective "attack softness" rather than a physical measurement.

### 4.6 Manual adjustment

The **tuner** (Tuner tab of `scripts/app.py`, logic in `tuning.py`, window in `viz/tuner.py`) shows one take at a time: measured onset levels as bars, the calibrated model as markers, and the RMS error. It provides:

- sliders for finger width, $\sigma_0$, $\sigma_2$, and $B$, for the current string (width for the current finger type);
- a per-harmonic offset in dB (choose the harmonic, then set the offset), plus **Match harmonic**, which sets the offset that closes the gap for the selected harmonic;
- a pluck-position slider that only shifts the view, to see how the model responds; it is not saved;
- Previous / Next take, Clear string (drops that string's manual adjustments), A/B playback of the recording and the calibrated model (matched in loudness), and Save.

Adjustments go to the profile's manual section, which overrides but never replaces the fitted values. Refitting with `scripts/calibrate.py` replaces the fitted values, keeps every manual adjustment, and lists the ones it kept.

### 4.7 Guitar profile format

`calibration/<name>.json` (`profile.py`):

```json
{
  "name": "my_classical",
  "scale_length_mm": 650,
  "string_set": "daddario_ej46",
  "recorder": "iphone_voice_memos",
  "created": "2026-10-05",
  "source_measurements": ["E2_040mm_flesh_t1", "..."],
  "response_curve": {"freq_hz": [82.1, 83.0, "..."], "gain_db": [-3.2, -2.9, "..."]},
  "finger_width_mm": {"fitted": {"flesh": 11.5}, "manual": {}},
  "strings": {
    "E2": {
      "fitted": {"f0": 82.3, "inharmonicity_b": 2.1e-5, "sigma0": 0.9, "sigma2": 1.4e-6,
                 "sigma_n": [0.95, 1.1, null, "..."]},
      "manual": {"offsets_db": {"7": -2.0}}
    }
  }
}
```

**Using a profile.** Pass `ModelOptions(profile=...)`, which takes a name (looked up in `calibration/`), a path, or a loaded `GuitarProfile`. The profile supplies:
- tuning ($f_0$), scale length, damping (measured $\sigma_n$ where available, unless $\sigma_0$ or $\sigma_2$ was adjusted by hand), and $B$;
- the finger width for the pluck's finger type;
- $H(f)$ as a gain on each partial, plus manual per-harmonic offsets.

The `inharmonicity` and `body` layers default to automatic: on when a profile provides the values. An explicit `True` or `False` still wins. Strings missing from the profile keep their defaults. Scripts take `--profile`.

---

## 5. Architecture

### 5.1 Project layout

```
SpectrumModeler/
├── environment.yml          # conda environment
├── pyproject.toml           # package metadata, ruff and pytest configuration
├── README.md
├── .gitignore
├── docs/
│   ├── design.md            # this document
│   ├── roadmap.md           # stages and progress checklist
│   └── recording-guide.md   # how to record; iPhone concerns
├── src/spectrum_modeler/
│   ├── __init__.py
│   ├── data/string_sets/    # string-set TOML files (daddario_ej46.toml)
│   ├── string_sets.py       # load string sets; lbf -> N
│   ├── params.py            # dataclasses: StringParams, PluckParams, ModelOptions
│   ├── presets.py           # open strings, pluck positions, finger types
│   ├── modal.py             # a_n, observables, damping, width, inharmonicity
│   ├── synthesis.py         # additive synthesis -> numpy audio buffer
│   ├── body.py              # illustrative body resonances (uncalibrated fallback)
│   ├── synthetic.py         # generate fake sessions with known parameters
│   ├── recording.py         # decode (ffmpeg), manifest, onset segmentation, quality checks
│   ├── measure.py           # f0, harmonic peaks, decay rates, B -> Measurement
│   ├── calibrate.py         # joint least-squares fit -> GuitarProfile
│   ├── profile.py           # GuitarProfile: fitted + manual values, JSON load and save
│   ├── tuning.py            # tuner logic (no GUI), for testing
│   ├── explorer.py          # explorer logic (no GUI): settings, A/B pinning, audio
│   ├── compare.py           # two recordings aligned, loudness-matched, and measured alike
│   ├── analysis.py          # shared FFT and STFT helpers, spectral centroid
│   ├── audio.py             # WAV read and write, optional playback via sounddevice
│   ├── fdtd.py              # finite-difference string simulation (§2.4)
│   └── viz/
│       ├── spectrum.py      # harmonic bar charts, A/B comparison
│       ├── shape.py         # string shape and mode-sum overlays
│       ├── spectrogram.py   # spectrograms, single and side by side
│       ├── measured.py      # recording spectrum, model overlay
│       ├── calibration.py   # H(f) plot, fit residuals, beta check
│       ├── tuner.py         # manual adjustment window (§4.6)
│       ├── explorer.py      # explorer window (§6)
│       ├── animate.py       # string animation, filmstrip, modal check (§2.4)
│       ├── app.py           # the tabbed app window (§6)
│       └── style.py         # light and dark themes, matplotlib settings
├── scripts/
│   ├── compare_positions.py # A/B plots and WAVs for two pluck positions
│   ├── make_synthetic_session.py # fake session for testing the pipeline
│   ├── import_recordings.py # decode, check manifest, segment
│   ├── analyze.py           # measure a session, or any single file (ad-hoc); plot
│   ├── calibrate.py         # fit profile from measurements
│   ├── app.py               # launch the app: explorer, compare, tuner
│   ├── new_manifest.py      # write a session's manifest before recording
│   ├── animate_string.py    # simulate and animate the string (§2.4)
│   └── compare_recordings.py # two recordings: spectrograms, difference, harmonics (§6)
├── recordings/              # real session directory
│   ├── manifest.csv         # committed
│   ├── raw/                 # iPhone .m4a files (git-ignored by default)
│   ├── processed/           # decoded clips, takes, import_report.json (git-ignored)
│   ├── measurements/        # per-clip JSON (committed)
│   └── plots/               # measurement plots (regenerable)
├── calibration/             # guitar profiles (committed)
├── tests/                   # pytest; synthetic sessions as fixtures (conftest.py)
└── output/                  # generated WAVs and figures (git-ignored)
```

**Data in version control:** the manifest, the measurements, and the profiles are small text files and are committed. Raw audio is git-ignored by default and backed up separately. It can be added with Git LFS later if you want the recordings versioned too.

### 5.2 Core data model

```python
@dataclass(frozen=True)
class StringParams:
    note: str                 # "E2" ... "E4"
    f0: float                 # Hz
    length: float = 0.650     # m
    tension: float | None = None        # N, from the string set
    diameter: float | None = None       # m, optional
    youngs_modulus: float | None = None # Pa, optional
    inharmonicity_b: float | None = None  # B directly, if known
    sigma0: float = 1.0                 # 1/s, damping
    sigma2: float = 2e-6                # 1/(s Hz^2), damping

    @property
    def linear_density(self) -> float | None:  # kg/m, mu = T / (2 L f0)^2
        ...

    @property
    def b_coefficient(self) -> float | None:  # inharmonicity_b, else from d, E, T
        ...

@dataclass(frozen=True)
class PluckParams:
    distance_mm: float        # from saddle; beta = distance / length
    amplitude: float = 0.002  # m
    width_mm: float = 0.0     # 0 = ideal point pluck
    finger: str | None = None # a profile may supply the width for this finger type

@dataclass(frozen=True)
class ModelOptions:
    n_harmonics: int = 40
    observable: Literal["displacement", "velocity", "bridge_force"] = "bridge_force"
    damping: bool = True
    finger_width: bool = True
    inharmonicity: bool | None = None  # None: on if the profile provides B
    body: bool | None = None           # None: on with a profile (calibrated H)
    profile: str | Path | GuitarProfile | None = None  # calibrated values
```

`measure.Measurement` holds one take: metadata (clip, take, note, pluck position, finger), $f_0$, $B$, $\sigma_0$, $\sigma_2$, off-series peaks, and per-harmonic arrays (frequency, window level, onset level, floor, decay rate, reliability).

### 5.3 Key functions

| Function | Returns |
|---|---|
| `modal.coefficients(string, pluck, opts)` | `ModalSpectrum` (harmonics, freqs, signed amps, decay_rates, plus the inputs), the single source of truth for the model spectrum |
| `modal.missing_harmonics(beta, n_max)` | Harmonic numbers with a node at the pluck point |
| `modal.envelope_nulls(beta, n_max)` | Where $\sin(n\pi\beta) = 0$ ($n = k/\beta$, not necessarily whole); marked on spectrum plots |
| `modal.initial_shape(x, length, pluck)` | Triangle at release, used for plots, tests, and the simulation |
| `modal.mode_contributions(x, string, pluck, n_modes)` | Each mode's shape at release; summed for partial-sum plots |
| `synthesis.render(string, pluck, opts, duration)` | Float audio array, peak-normalized by default. The step at release is smoothed over one cycle of the highest partial, matching the bandwidth of the rest of the waveform without blurring the first pulse |
| `audio.match_loudness(*signals)` | Signals at equal RMS under one shared peak, for fair A/B listening |
| `recording.import_session(session)` | `ImportReport`: clips, onsets, takes, and every issue found; also saved as JSON |
| `recording.detect_onsets(audio, sr)` | Sample indices of pluck onsets |
| `measure.measure(audio, sr, onset, f0_hint=...)` | `Measurement` (onset and $f_0$ detected if not given) |
| `measure.model_for(measurement)` | The uncalibrated model for comparison, or None if the note or position is unknown |
| `synthetic.generate_session(root, config)` | A synthetic session with `truth.json` |
| `measure.measure_clip(session, clip)` | `Measurement` for every take of an imported clip |
| `calibrate.calibrate(measurements, name=...)` | `CalibrationResult`: the `GuitarProfile` plus per-take fits and data for the report |
| `profile.load_profile(name_or_path)` | `GuitarProfile` (an independent copy; loads are cached until the file changes) |
| `tuning.TunerState(profile, measurements, path, session)` | Tuner logic: model vs take, adjustments, save, A/B audio |
| `analysis.spectral_centroid(freqs, amps)` | A single "brightness" number: the power-weighted mean frequency. Amplitude weighting would let a truncated 1/n series be dominated by its cut-off |
| `explorer.ExplorerState(profile=..., session=..., measurements=...)` | Explorer logic: current and pinned settings, spectra, readout, A/B and recording audio |
| `fdtd.simulate(string, pluck, duration, damping=..., stiffness=..., snapshot_every=...)` | `FdtdResult`: bridge force every step, displacement snapshots, optional energy |
| `recording.planned_rows(notes, date, ...)` | Manifest rows for a planned session, named as the recording guide says |

Design principle: the computational code (`modal`, `synthesis`, `measure`, `calibrate`, `fdtd`, `body`) is NumPy/SciPy with **no plotting**, and works on arrays rather than files. File access lives in `recording`, `profile`, `audio`, `synthetic`, and the session helpers at the end of `measure`; plotting lives in `viz`. This keeps the physics and the measurement code testable.

---

## 6. Visualizations

1. **Harmonic spectrum bar chart.** Amplitudes in dB against harmonic number. Small triangles along the top mark where the pluck point is a node ($n = k/\beta$), explained in a legend entry.
2. **A/B comparison.** Two plucks side by side, string shape above and spectrum below, on one dB scale (ponticello against tasto, for example). The explorer's readout gives the spectral centroid of each.
3. **String shape.** The initial triangle with the pluck point marked; optionally the first N mode shapes overlaid, plus their partial sum converging to the triangle.
4. **Measured spectrum.** The FFT of a recording, with detected harmonic peaks and the local noise floor marked.
5. **Model against recording.** Measured harmonic levels (bars) overlaid with the uncalibrated model (markers); the tuner shows the calibrated model the same way.
6. **Spectrogram.** Of a recording or of synthesized audio, showing high harmonics decaying faster.
7. **Calibration report.** The fitted $H(f)$ curve, the per-take errors, and the fitted β against the taped position.
8. **Explorer** (a tab of the app, item 13). String and finger selectors, a pluck-position slider, realism-layer checkboxes, a calibrated/uncalibrated switch, and a readout of β, the spectral centroid, missing harmonics, and the nearest recording. **Pin as B** freezes the current settings and overlays them on the spectrum. **Play model**, **Play recording**, and **Play A then B** (matched in loudness) are the playback buttons. Artists are updated in place: about 1 ms per change plus the redraw.
9. **Brightness against position.** The spectral centroid (÷ $f_0$) against pluck distance, one line per finger type. In the uncalibrated model it is the same for every open string.
10. **Tuner** (a tab of the app, item 13). See §4.6.
11. **Recording comparison** (`scripts/compare_recordings.py`, logic in `compare.py`). Two recordings, each starting at a chosen pluck and running to the next, trimmed to the same length and matched in loudness (RMS). The panels:
    - both spectrograms on one dB scale;
    - a **difference map**, B − A in dB on a diverging scale (A's blue through neutral grey to B's orange, ±20 dB). It is blanked where neither recording is 15 dB above its *own* noise floor, because loudness matching raises the quieter recording's noise;
    - harmonic levels at the pluck (from the standard measurement) for both;
    - the decay of up to four chosen harmonics.

    A literal overlay of two spectrograms is not used: blended colour maps make both unreadable.
12. **Animation** (`scripts/animate_string.py`). String motion over time from the finite-difference solver, in slow motion with the bridge force below, exported as MP4 or GIF; plus a one-period filmstrip and a check against the modal model.
13. **The app** (`scripts/app.py`, window in `viz/app.py`). One Tk window with tabs: Explorer (item 8), Compare (item 2, spectrograms, and mode decompositions, for any two positions), and Tuner (§4.6). The Explorer and Tuner tabs are the matplotlib windows, embedded; Compare has Tk controls in a sidebar. A bar above the tabs opens an analyzed session and a guitar profile (`Workspace`, `load_workspace`); opening either rebuilds the tabs, and both are remembered between runs in `~/.config/spectrum-modeler/app.json`. Whatever fails to open is reported in the bar and forgotten. The app draws in the dark theme (`viz/style.py`: the same hues stepped for a dark surface); exports and the scripts use the light theme. Notes on Tk:
    - TkAgg draws at 1× on Retina displays, so charts are softer than in a native matplotlib window. Qt (PySide6) would fix this, at the cost of a large dependency.

---

## 7. Environment and Engineering Practices

### 7.1 Conda environment

Dependencies are in `environment.yml` (conda-forge, including ffmpeg for decoding and animation export); the package is installed into it in editable mode. Setup commands are in the README.

### 7.2 Practices

- **Version control:** git, with small focused commits.
- **Code quality:** `ruff check` and `ruff format`, type hints on all public functions, and docstrings stating units (Hz, mm, dB).
- **Tests:** `pytest` runs before every commit. Physics and measurement code needs tests before it is used in plots.
- **Reproducibility:** plots and profiles are generated by scripts, not assembled by hand.
- **Documentation:** this design document is kept current. When the plan changes, update this document, then the code. The roadmap tracks progress.

---

## 8. Testing Strategy

| Test | Verifies |
|---|---|
| Sum of $a_n \sin(n\pi x/L)$ for N = 2000 reconstructs the triangle (max error below 0.5 % of h) | Coefficient formula |
| $a_n$ matches a numerical sine transform of `initial_shape` | Formula against an independent method |
| $a_n \approx 0$ for every $n$ with $n\beta$ a whole number | Missing-harmonic logic |
| $a_n(\beta) = (-1)^{n+1} a_n(1-\beta)$ | Mirror symmetry of the string |
| Bridge-force spectrum decays as ~1/n on average; displacement as ~1/n² | Observables |
| Rendered audio peaks at 1.0 and is finite; decay envelope matches $\sigma_n$ | Synthesis |
| **Round trip:** synthesize a note with known β, $w$, $\sigma_0$, $\sigma_2$, and $B$, add noise, then measure it. The recovered values match the inputs within tolerance | `measure.py` |
| **Round trip:** synthesize a set of takes through a known $H(f)$, then calibrate. The fitted $H(f)$ and $w$ match the known values | `calibrate.py` |
| Onset detection finds synthetic plucks within 5 ms; clipping is detected | `recording.py` |
| **End to end:** generate a synthetic session → import → measure → calibrate. No errors; every injected impairment is flagged; fitted values match the known ones | Whole pipeline |
| Ad-hoc analysis of a non-guitar file (noise, a chord, silence) returns a result or a clear error, never a crash | `measure.py`, `analyze.py` |
| Profile save then load gives identical values | `profile.py` |
| Finite-difference simulation conserves energy within tolerance with damping off; its FFT matches the modal result | Optional simulation stage |

The round-trip tests matter most. They allow the measurement and calibration code to be trusted before it ever sees a real recording.

---

## 9. Risks and Open Questions

- **Recording quality (iPhone first pass).** Concerns about the phone's signal processing, microphone response, room reflections, pluck variability, and two-stage decay are covered in [recording-guide.md §4](recording-guide.md#4-concerns). That guide also sets the criteria for deciding whether iPhone recordings are good enough (§5 there). The fallback is a better recorder; the pipeline does not change.
- **Loudness bias in A/B listening.** A brighter sound can seem louder. Normalize loudness (RMS) for A/B playback.
- **Missing string data.** The product page gives tensions but not diameters, so $B$ comes from calibration, not first principles. This is acceptable: the fitted value is the more trustworthy one.

---

## 10. Possible Future Extensions

- **Fretted notes:** a shorter vibrating length, with the pluck position still measured from the saddle.
- Nail and flesh comparison as a first-class feature, with profiles for each finger type.
- Two-polarization modeling (vertical and horizontal) for more realistic decay.
- Comparing pluck angle and the attack of a rest stroke with a free stroke.
- A browser version (JavaScript with the Web Audio API) reusing the modal formulas and exported profiles.
- Other instruments, such as steel-string guitar, harp, and harpsichord, by changing presets and recording new profiles.
