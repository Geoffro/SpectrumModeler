# SpectrumModeler

Models a plucked classical guitar string to show how the **pluck position** changes its **overtones** and so its timbre: bright and nasal near the bridge (*ponticello*), warm and round over the fingerboard (*tasto*).

It computes the theoretical harmonic spectrum, synthesizes audio, measures recordings, and calibrates the model against your guitar.

## Setup

Requires [conda](https://docs.conda.io/en/latest/miniconda.html).

```bash
conda env create -f environment.yml
conda activate spectrum-modeler
pytest -m "not slow"     # quick check (the full suite also runs the scripts)
```

After changing `environment.yml`:

```bash
conda env update -f environment.yml --prune
```

## Quick start

```bash
python scripts/explore.py                 # interactive: drag the pluck point, hear the result
python scripts/compare_positions.py       # plots and WAVs, ponticello vs tasto, all six strings
python scripts/animate_string.py          # slow-motion animation of the string itself
```

`compare_positions.py` writes to `output/`: spectra, spectrograms, one WAV per pluck, and an A-then-B WAV matched in loudness. Options: `--string E2`, `--a 40 --b twelfth_fret` (preset or mm), `--finger nail`, `--no-damping`, `--body`, `--inharmonicity 5e-5`, `--profile my_guitar`, `--detail` (mode decompositions).

**Explorer** controls: string, finger, realism layers, calibrated/uncalibrated, and the pluck-position slider. **Pin as B** freezes the current settings for comparison. **Play A then B** plays both, matched in loudness. **Play recording** plays the recorded take nearest the current position (needs `--session`).

## Calibrating to your guitar

1. **Record** open strings at marked positions, following [docs/recording-guide.md](docs/recording-guide.md). Put the files in `recordings/raw/` and list them in `recordings/manifest.csv`.
2. **Import:** decode, check against the manifest, split into one take per pluck, and flag problems.
   ```bash
   python scripts/import_recordings.py --session recordings
   ```
3. **Measure** every take (pitch, harmonic levels, decay, stiffness); plots go to `recordings/plots/`.
   ```bash
   python scripts/analyze.py recordings
   ```
4. **Calibrate:** fit the body/room/mic response, finger width, and per-string damping and stiffness into `calibration/my_guitar.json`, with a report plot.
   ```bash
   python scripts/calibrate.py --session recordings --name my_guitar
   ```
5. **Fine-tune** by hand (optional). Adjustments are kept when you recalibrate.
   ```bash
   python scripts/tune.py --session recordings --profile my_guitar
   ```
6. **Explore** with the calibrated model:
   ```bash
   python scripts/explore.py --profile my_guitar --session recordings
   ```

**No recordings yet?** Run the same pipeline on a synthetic session with known answers:

```bash
python scripts/make_synthetic_session.py --force                 # add --impair all to test the checks
python scripts/import_recordings.py --session output/synthetic_session
python scripts/analyze.py output/synthetic_session
python scripts/calibrate.py --session output/synthetic_session --name synthetic --profiles output/calibration
```

**Compare two recordings:** spectrograms side by side on one scale, a B − A difference map, harmonic levels at the pluck, and the decay of individual harmonics. Each recording is aligned at its pluck and matched in loudness; an A-then-B WAV is written too.

```bash
python scripts/compare_recordings.py a.m4a b.m4a --note E2
python scripts/compare_recordings.py recordings/processed/E2_040mm_flesh_t1.wav recordings/processed/E2_200mm_flesh_t1.wav
python scripts/compare_recordings.py same_clip.m4a same_clip.m4a --pluck-b 2    # two plucks of one clip
```

**Any single audio file** (MP3, M4A, WAV, …) can be measured without a manifest:

```bash
python scripts/analyze.py some_note.mp3                  # pitch detected
python scripts/analyze.py my_E2.m4a --note E2 --pluck-mm 130   # adds the model for comparison
```

## Strings

The default string set is D'Addario Pro-Arté Hard Tension (EJ46). To use other strings, add a TOML file to `src/spectrum_modeler/data/string_sets/` and pass `--string-set <name>`.

## Development

```bash
pytest              # all tests (about a minute)
pytest -m "not slow"
ruff check .        # lint
ruff format .       # format
```

## Documentation

- [Design](docs/design.md): physics, architecture, measurement and calibration methods
- [Roadmap](docs/roadmap.md): stages and progress
- [Recording guide](docs/recording-guide.md): how to record the guitar for calibration
