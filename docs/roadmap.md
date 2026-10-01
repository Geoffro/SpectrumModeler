# SpectrumModeler — Roadmap

Design details for every stage are in [design.md](design.md).

All code (Stages 0–5) is built and tested before any recording is made. Until then, the recording pipeline is exercised with a **synthetic session** (generated clips with known parameters) and with **arbitrary audio files** (MP3s and the like) in ad-hoc mode. Stage 6 is the first time real recordings are used.

## Stage 0 — Project setup

- [x] add `.gitignore` (covering `output/`, `recordings/raw/`, `recordings/processed/`, `__pycache__/`)
- [x] `environment.yml` (design §7.1); run `conda env create` successfully
- [x] `pyproject.toml` with package metadata, ruff configuration, and pytest configuration
- [x] Empty package `src/spectrum_modeler/` installed in editable mode
- [x] `README.md` explaining the project and how to set it up
- [x] Placeholder test runs green under `pytest`

**Done when:**
- [x] A fresh `conda env create` followed by `pytest` passes
- [x] `ruff check` is clean

---

## Stage 1 — Analytic spectrum

- [x] `params.py`: `StringParams`, `PluckParams`, `ModelOptions`
- [x] `presets.py`: six open strings, pluck-position presets, finger types
- [x] `string_sets.py` and `data/string_sets/daddario_ej46.toml` (tensions in lbf, converted to N; derived $\mu$)
- [x] Test: bundled string set loads; the derived $\mu$ reproduces each $f_0$
- [x] `modal.coefficients` for the ideal string, all three observables
- [x] `modal.missing_harmonics`, `modal.initial_shape`
- [x] Tests: triangle reconstruction, numerical sine transform, missing harmonics, mirror symmetry, roll-off rates (design §8)
- [x] `viz/spectrum.py`: bar chart with missing harmonics marked; A/B comparison
- [x] `viz/shape.py`: initial shape, mode overlays, partial sums
- [x] `scripts/compare_positions.py` writes a PNG to `output/`

**Done when:**
- [x] All Stage 1 tests pass
- [x] The β = 0.5 plot shows every even harmonic at zero
- [x] The ponticello and tasto comparison is generated for all six strings

---

## Stage 2 — Audio synthesis

- [x] `synthesis.render` (additive synthesis) with the damping layer ($\sigma_0$, $\sigma_2$)
- [x] Finger-width layer ($S_n$) and inharmonicity layer ($B$), each switchable
- [x] Fallback body filter in `body.py` (illustrative biquad resonances)
- [x] `audio.py`: WAV write and read, playback via sounddevice
- [x] `analysis.py`: STFT and spectrogram helpers, spectral centroid
- [x] `compare_positions.py` also writes `output/<note>_<preset>.wav`
- [x] Tests: synthesis output (finite, normalized, decay matches $\sigma_n$)

**Done when:**
- [ ] The ponticello and tasto WAVs sound clearly different on every string
- [x] A spectrogram of a rendered note shows the high harmonics decaying first
- [ ] Switching each realism layer on and off produces an audible and visible change

---

## Stage 3 — Import and measurement

### 3a. Synthetic session generator
- [x] `synthetic.py`: generate a fake session (clips with 3 plucks each, a room-tone clip, and a manifest) from known β, $w$, $\sigma_0$, $\sigma_2$, $B$, $H(f)$, per-take gain, and noise level
- [x] Optional impairments, for testing the checks: clipping, a lossy codec, a missing manifest row, sympathetic peaks
- [x] `scripts/make_synthetic_session.py` writes a session directory (default `output/synthetic_session/`)

### 3b. Import pipeline
- [x] `recording.py`: ffprobe codec check (warn on lossy), ffmpeg decode to 48 kHz mono float WAV, cached
- [x] Accepts `.m4a`, `.wav`, `.flac`, `.aiff`, `.mp3`
- [x] Manifest check (files missing from the manifest, rows with no file, detected pluck count ≠ `plucks`)
- [x] Onset detection; each clip split into takes `t1`, `t2`, `t3`
- [x] Quality checks: clipping, low level, noise floor compared with room tone
- [x] `scripts/import_recordings.py --session <dir>`
- [x] Tests: onset detection, clipping detection, manifest mismatches (all on synthetic sessions)

### 3c. Measurement
- [x] `measure.py`: f0 refinement, harmonic peak picking, noise floor, reliability mask, flags for peaks off the harmonic series
- [x] f0 auto-detection when no note is given (ad-hoc mode)
- [x] Decay rate per harmonic from the STFT
- [x] Estimate of $B$ from the partial frequencies
- [x] Measurements saved to `<session>/measurements/<clip>.json`
- [x] **Round-trip test:** synthesized note → measure → recovers $\sigma_0$, $\sigma_2$, and $B$
- [x] `viz/measured.py`: spectrum with peaks marked, spectrogram, uncalibrated model overlay
- [x] `scripts/analyze.py`: session mode, plus ad-hoc mode for any single file (`--note` / `--f0` optional)

**Done when:**
- [x] The round-trip measurement test passes
- [x] A synthetic session imports, measures, and plots with no errors, and every injected impairment is flagged
- [x] Ad-hoc analysis of an arbitrary MP3 produces a spectrum plot without crashing (single notes give sensible harmonics; anything else at least degrades gracefully)

---

## Stage 4 — Calibration

- [x] `profile.py`: `GuitarProfile` dataclass, JSON save and load, with test
- [x] `calibrate.py`: joint least-squares fit of $G$ per take, $H(f)$ spline, and $w$
- [x] Per-string fit of $\sigma_0$, $\sigma_2$, and $B$; raw measured $\sigma_n$ kept
- [x] Optional refinement of β within ±5 mm, reported against the manifest position
- [x] `ModelOptions(profile=...)` switches on the calibrated layers in `modal` and `synthesis`
- [x] `viz/calibration.py`: $H(f)$ plot, per-take errors, β check
- [x] `viz/tuner.py`: sliders, per-harmonic offsets, A/B playback, save
- [x] `scripts/calibrate.py`

**Done when:**
- [x] **Round-trip test:** a synthetic session with a known $H(f)$ → the fit recovers $H$, $w$, and the decay parameters within tolerance
- [x] A profile calibrated from the synthetic session loads and changes the model output
- [x] The tuner saves manual offsets without overwriting fitted values

---

## Stage 5 — Explorer app

- [x] `viz/explorer.py`: string selector, pluck-position slider, layer checkboxes, calibrated/uncalibrated switch
- [x] Live spectrum, string shape, and centroid readout
- [x] Play buttons for the model and for the nearest recording
- [x] A/B playback with loudness normalization
- [x] `scripts/app.py`: explorer, comparison, and tuner as tabs in one dark-themed window
- [x] README updated with usage instructions

**Done when:**
- [x] The slider updates the plots without noticeable delay
- [x] Every open string works in uncalibrated mode, and in calibrated mode with the synthetic profile

---

## Stage 6 — Real recordings and calibration

Follow [recording-guide.md](recording-guide.md).

### 6a. Pilot session (E2 and G3)
- [ ] One-time setup done (recording guide §1)
- [ ] Scale-length default updated if it is not 650 mm
- [ ] Manifest written: `python scripts/new_manifest.py --date <date> --notes E2 G3 --mic-distance <cm> --scale-length <mm>`
- [ ] Room tone + 10 clips recorded, renamed, and transferred to `recordings/raw/`
- [ ] Imported and measured

### 6b. Pilot review (recording guide §5)
- [ ] 1. Every clip is ALAC
- [ ] 2. No clipping warnings
- [ ] 3. Noise floor between plucks within ~3 dB of room tone, and steady
- [ ] 4. Takes agree within ~2 dB for harmonics 1–10
- [ ] 5. At 40 mm: E2 has 15 or more usable harmonics; G3 has 10 or more
- [ ] 6. 12th fret: harmonics 2, 4, 6 at least 10 dB below their neighbors
- [ ] 7. 130 mm: dip near harmonic 5
- [ ] 8. Decay roughly straight in dB over the first 2 s
- [ ] **Decision recorded:** continue with iPhone / fix and redo pilot / upgrade recorder (reasons written here)

### 6c. Full session and calibration
- [ ] All 6 strings × 5 positions (30 clips + room tone) recorded with the pilot setup
- [ ] Imported and measured without errors
- [ ] First real profile committed: `calibration/<guitar_name>.json`

**Done when:**
- [ ] The measured spectra show the pattern of missing harmonics moving with pluck position
- [ ] The calibrated model matches reliable measured harmonics (up to harmonic 15) within ~3 dB RMS for most takes
- [ ] The fitted β values are within ±5 mm of the taped positions (or the outliers are explained)
- [ ] In blind A/B listening, the calibrated model sounds noticeably closer to the recording than the uncalibrated one

---

## Stage 7 — Time-domain simulation (optional)

- [x] `fdtd.simulate` with loss terms and a check of the Courant condition
- [x] Energy conservation test with damping off
- [x] `viz/animate.py`: string motion animation, MP4 or GIF export

**Done when:**
- [x] An FFT of the simulated bridge force matches the modal $F_n$ within 1 dB for the first 20 harmonics
- [x] Animation exported for ponticello and tasto plucks

---

## Backlog (after version 1)

- [ ] Nail recording session and a nail profile
- [ ] Re-record with a better recorder (if the iPhone passed) and compare profiles
- [ ] Fretted notes
- [ ] Two-polarization decay model
- [ ] Rest stroke compared with free stroke
- [ ] Browser version
