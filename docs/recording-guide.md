# SpectrumModeler — Recording Guide

iPhone recording procedure for calibration. Processing details: [design.md §4](design.md#4-recordings-and-calibration).

**Rule:** consistency beats quality. The calibration absorbs a colored mic or room, but not a setup that changes between clips.

---

## 1. Setup (once)

**Gear:** iPhone, stand, steel ruler, painter's tape, card strip (~25 × 2 cm), cloth for muting, tuner.

**iPhone**
- Settings → Apps → Voice Memos → Audio Quality → **Lossless**
- Never use *Enhance Recording*. If Control Center offers Mic Mode, choose **Standard**.
- Airplane Mode + Do Not Disturb while recording.

**Guitar**
- Strings settled (not freshly installed).
- Measure scale length (nut to saddle top) on the 1st and 6th strings. Record both.
- 12th-fret position = half the scale length.

**Pluck guide**
- Mark the card at **40, 80, 130, 200 mm**. Tape it beside the 1st string, zero end at the saddle.
- Fifth position: pluck directly over the **12th fret**.

**Placement**
- Phone 30–50 cm in front of the soundhole, at soundhole height. Same orientation every time.
- Tape the chair and stand positions on the floor. Photograph the setup; note the distance.

---

## 2. Each Session

0. **Before recording**, write the manifest and get the list of clip names:
   ```bash
   python scripts/new_manifest.py --date 2026-10-04 --notes E2 G3 --mic-distance 40 --scale-length 650
   ```
   Use your measured scale length; it sets the 12th-fret `pluck_mm`. For the full session, run it again with `--notes all --append`.
1. Quiet room: fans and HVAC off.
2. Tune.
3. **Room tone:** 10 s of silence, seated in position → `ROOMTONE_<date>`.
4. **Level check:** firm pluck on the low E at the 12th fret. If the waveform hits the edges, move the phone back 10 cm and note the new distance. Delete the clip.
5. For each string, for each position:
   - Mute the other strings (cloth near the nut).
   - New memo → wait 2 s → **3 plucks** (index finger, free stroke, flesh, mezzo-forte). Before each next pluck: let it ring 8 s → damp with a fingertip → 2 s silence.
   - Stop. Rename the memo right away: `E2_040mm_flesh`.
   - Note any bad pluck. Redo the clip if more than one pluck is bad.

| Session | Strings | Clips |
|---|---|---|
| Pilot | E2, G3 | 10 + room tone |
| Full | all 6 | 30 + room tone |
| Nail (optional) | all 6 | 30 + room tone |

---

## 3. After the Session

1. AirDrop the clips to the Mac → `recordings/raw/`.
2. Check the filenames: `<note>_<pos>mm_<finger>.m4a`, positions zero-padded (`040`). Use `325` for the 12th fret.
3. Update `recordings/manifest.csv` where the session differed from the plan: a moved phone (`mic_distance_cm`), a bad pluck (`notes`), or a pluck count that isn't 3.
4. `python scripts/import_recordings.py`. The codec should be **ALAC**; fix any warnings. Then `python scripts/analyze.py recordings`.

---

## 4. Concerns

| Concern | Symptom in the data | Response |
|---|---|---|
| Automatic gain | Noise floor between plucks louder than room tone, or rising; all harmonics bend at the same moment | Per-take gain absorbs constant changes; otherwise use a measurement-mode app (§6) |
| Noise reduction | Harmonics drop suddenly into the floor; floor unnaturally smooth; holes between harmonics | Mic Mode Standard; otherwise a measurement-mode app |
| Lossy compression | Codec shows `aac` | Set Lossless *before* recording; earlier clips stay compressed |
| Mic response / weak lows | $H(f)$ falls off below ~100 Hz | Absorbed by $H(f)$. The profile then describes this guitar *as heard by this phone* |
| Room reflections | Fine ripple in $H(f)$ that changes when the phone moves 10 cm | Furnished room, away from walls; smoothness penalty in the fit |
| Placement sensitivity | Large changes from small moves | 30–50 cm; floor marks; photo |
| Pluck variability | Takes differ by more than a few dB | Practice the stroke; 3 takes; exclude outliers |
| Sympathetic strings | Peaks that are not harmonics of the recorded string | Mute the other strings; the measurement code flags them |
| Two-stage decay | Decay in dB bends instead of running straight | Accept for v1; report the misfit |
| Clipping / handling noise | Clipping warning; low thumps | Level check; don't touch the phone; quiet clothing |

---

## 5. Pilot Review

Run on the imported and measured pilot session. Tracked in the [roadmap](roadmap.md).

| # | Check | Pass |
|---|---|---|
| 1 | Codec | All ALAC |
| 2 | Clipping | None |
| 3 | Noise floor | Within ~3 dB of room tone; steady |
| 4 | Take agreement | Within ~2 dB, harmonics 1–10 |
| 5 | Usable harmonics at 40 mm | E2 ≥ 15, G3 ≥ 10 (more than 10 dB above the floor) |
| 6 | 12th fret | Harmonics 2, 4, 6 ≥ 10 dB below their neighbors |
| 7 | 130 mm | Dip near harmonic 5 |
| 8 | Decay | Roughly straight in dB over the first 2 s |

All pass → full session. Failures → apply the fixes in §4 and redo the pilot. Checks 3, 5, 6, or 8 still failing → §6.

The thresholds are initial estimates; adjust them after seeing real data.

---

## 6. If the iPhone Isn't Good Enough

In order of effort:
1. An iPhone recording app that uses iOS *measurement* mode (no gain control or noise reduction) and exports WAV.
2. An external USB-C or Lightning microphone for the iPhone.
3. An audio interface + small-diaphragm omni condenser, recording 24-bit/48 kHz WAV on the Mac.

Each time: set `recorder` in the manifest, redo the pilot, and create a new profile. Keep the iPhone profile for comparison.
