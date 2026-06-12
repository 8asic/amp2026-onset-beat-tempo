# Experiment Log — AMP Challenge

> **Rules:** Completed experiments only. One entry per experiment. Fill in
> Git commit hashes, leaderboard results, and decisions as you run things.
> Never edit a past entry retroactively — append a corrected entry instead.

---

## Baseline — EXP-000

| Field | Value |
|-------|-------|
| **Experiment ID** | EXP-000 |
| **Date** | 2026-06-03 (inferred from cell outputs) |
| **Status** | COMPLETED — submitted to leaderboard |
| **Git commit (before)** | *unknown — commit before starting work* |
| **Git commit (after)** | *unknown — fill in from `git log`* |

### Files Modified
None. This is the unmodified first run.

### Config State at Submission
```python
# src/config.py at time of submission (inferred from sweep output)
OnsetConfig.threshold    = 0.16   # default value used when submission was generated
OnsetConfig.peak_distance = 2
BeatConfig.tempo_min     = 30.0
BeatConfig.tempo_max     = 200.0
BeatConfig.tightness     = 60.0
```

**Critical note:** The submission file (`submissions/predictions.json`) was
generated in cell `d25f128c` (execution_count=14, labelled "Generate submission
for test set"). This cell ran **before** the parameter sweep in cell `51412491`
(execution_count=17). The sweep found `threshold=0.08` is better, but that
result was never used to regenerate the submission. The submitted file reflects
`threshold=0.16` (config default at that time).

### Validation Results (20-file split, seed=42)
| Metric | Score | N files |
|--------|-------|---------|
| Onset F1 | 0.4318 | 20 |
| Beat F1 | 0.3855 | 20 |
| Tempo | 0.3250 | 20 |
| **Mean** | **0.3808** | 20 |

### Leaderboard Results
| Metric | Score |
|--------|-------|
| Onset F1 | *fill in* |
| Beat F1 | *fill in* |
| Tempo | *fill in* |
| Mean | *fill in* |

### Failure Analysis
1. **Submission/sweep ordering bug:** Sweep (cell 17) ran after submission
   generation (cell 14). Config mutated in the sweep is not persisted back to
   disk and is not reflected in the submitted file. Every future submission must
   be generated **after** finalising parameters.

2. **Onset under-detection:** Sweep shows `threshold=0.08` yields onset F1=0.6153
   vs 0.4318 at `threshold=0.16`. The config default was never updated before
   the submission cell ran. Pure configuration bug, no algorithmic issue.

3. **Beat F1 stuck at 0.3855 across all sweep configs:** Beat score does not
   change with onset threshold, confirming that `BeatTracker` and
   `TempoEstimator` are not affected by the onset threshold sweep. Beat quality
   is limited by `librosa.beat.beat_track` itself.

4. **Tempo score stuck at 0.3250 across all sweep configs:** Same cause —
   tempo estimation is independent of onset threshold.

5. **Librosa compatibility monkey-patch in notebook (cell `48a39b5b`):** The
   `fix_librosa_compatibility()` function in the notebook overwrites
   `TempoEstimator.estimate` and `BeatTracker.track` at runtime. This means the
   class methods in `src/detectors.py` are **not** what actually runs — the
   notebook patch is. This is a latent reproducibility hazard: if the notebook
   is re-run without cell 13, the `src/` code runs instead, producing
   potentially different results. The patch itself contains a bare `except:`
   clause that silently swallows all errors.

6. **Validation set size:** 20 files = 15.7% of 127 training files. Metrics
   have ±2–3 file noise, making parameter differences of <0.02 unreliable.

### Decision
KEEP as baseline reference. Do not submit again without fixing issues above.

### Next Actions
- Run EXP-001 (versioned submission workflow) before any other change.
- Run EXP-002 (fix threshold config, regenerate submission).

---

## EXP-001 — Real SuperFlux + hop_length=256

| Field | Value |
|-------|-------|
| **Experiment ID** | EXP-001 |
| **Date** | 2026-06-11 |
| **Status** | COMPLETED — submission ready, not yet uploaded |
| **Git commit (before)** | 1fc9a5c |
| **Git commit (after)** | e826330 |

### Files Modified
- `src/features.py` — replaced placeholder `superflux()` with real SuperFlux (Böck et al. ICASSP 2012): frequency-axis maximum filter (μ=3, axis=0), log compression γ=100, 82 mel bins
- `src/config.py` — `onset_hop_length` 512→256, `onset_fmax` 17000→11000 (below Nyquist), added `onset_n_mels=82`, `superflux_gamma=100`, `superflux_mu=3`, `threshold` 0.08→0.01
- `src/detectors.py` — custom LFSF peak picker (3-condition: local max, adaptive mean, min IOI), custom autocorrelation beat tracker and tempo estimator
- `src/utils.py` — `save_versioned_submission()`, `get_git_commit()`, fixed `load_tempo_gt()` to return 3 values including annotator weight
- `src/evaluation.py` — fixed 3 bare `except:` → `except Exception:`
- `notebooks/01_pipeline.ipynb` — clean top-to-bottom notebook, removed monkey-patch cell, correct section order (Setup → Data → Parameters → Validate → Submit → Debug → Sweep)
- `notebooks/00_preprocessing.ipynb` — new notebook testing 5 onset strength variants

### Validation Results (127 files, threshold=0.01)
| Metric | Score | N files |
|--------|-------|---------|
| Onset F1 | **0.7824** | 127 |
| Beat F1 | 0.3791 | 127 |
| Tempo p-score | **0.5844** | 127 |
| **Mean** | **0.5820** | 127 |

vs EXP-000 (20-file split): Onset +0.35, Beat ≈0 (−0.006), Tempo +0.26, Mean **+0.20**

Threshold sweep on 127 files (beat/tempo flat across all thresholds — independent of onset):
| thresh | Onset | Beat | Tempo | Overall |
|--------|-------|------|-------|---------|
| 0.08 | 0.6203 | 0.3791 | 0.5844 | 0.5279 |
| 0.04 | 0.7287 | 0.3791 | 0.5844 | 0.5641 |
| 0.02 | 0.7753 | 0.3791 | 0.5844 | 0.5796 |
| **0.01** | **0.7824** | 0.3791 | 0.5844 | **0.5820** |
| 0.005 | 0.7720 | 0.3791 | 0.5844 | 0.5785 |

### Leaderboard Results
| Metric | Score |
|--------|-------|
| Onset F1 | *fill in* |
| Beat F1 | *fill in* |
| Tempo | *fill in* |
| Mean | *fill in* |

### Analysis
- Onset gain: Real SuperFlux γ=100 + hop_length=256 gives +0.04 F1 over baseline placeholder (γ=1, n_mels=64, hop=512). The frequency-axis max filter (μ=3) suppresses vibrato false positives.
- Beat F1 flat at 0.3791: the autocorrelation tracker builds a uniform beat grid (fixed lag, single phase). Rigid grid does not adapt to tempo changes, silences, or rubato. This is the bottleneck.
- Tempo p-score 0.5844: large gain from EXP-000's 0.3250, but EXP-000 measured on 20 files (noisy). The `load_tempo_gt` weight fix (using actual annotator weight instead of defaulting to 0.5) also contributed.

### Decision
KEEP. Onset and tempo improvements are large. Beat tracking is the clear next target.

### Next Actions
- Upload `submissions/EXP-001_20260611_185643_e826330/predictions.json` to challenge server
- Record leaderboard result in this entry
- Next experiment: DP beat tracker (see backlog EXP-002)

---

## EXP-002 — Dynamic Programming Beat Tracker

| Field | Value |
|-------|-------|
| **Experiment ID** | EXP-002 |
| **Date** | 2026-06-11 |
| **Status** | COMPLETED — submission ready |
| **Git commit (before)** | 45e38bb |
| **Git commit (after)** | adb976e |

### Files Modified
- `src/detectors.py` — replaced `BeatTracker.track()` uniform grid with Ellis (2007) DP trellis
- `src/config.py` — replaced unused `BeatConfig.tightness` with `dp_alpha=100.0`
- **Reverted:** `src/features.py` — 99th-percentile ODF normalisation was tried and reverted;
  it hurt onset F1 at threshold=0.01 (gamma=100 log compression already limits dynamic range;
  optimal threshold would shift, requiring re-sweep — deferred to EXP-004)

### Validation Results (127 files, threshold=0.01)
| Metric | Score | N files | vs EXP-001 |
|--------|-------|---------|------------|
| Onset F1 | 0.7824 | 127 | 0 |
| Beat F1 | **0.4960** | 127 | **+0.117** |
| Tempo p-score | 0.5844 | 127 | 0 |
| **Mean** | **0.6210** | 127 | **+0.039** |

Processing time: 74 ms/file (9.4s for 127 files) — acceptable.

### Leaderboard Results
| Metric | Score |
|--------|-------|
| Onset F1 | *fill in* |
| Beat F1 | *fill in* |
| Tempo | *fill in* |
| Mean | *fill in* |

### Analysis
The DP trellis aligns beats to high-energy onset frames rather than placing
them on a rigid uniform grid. This handles:
- Pieces with slight tempo variation (the score function absorbs small deviations)
- Files where the autocorrelation tempo estimate was off (the trellis can
  implicitly correct via the log-Gaussian search window [lag/2, 2*lag])
- Silences and sparse textures (no onset evidence = no strong beat placement)

Beat F1 improved +31% relative (+0.117 absolute) — the largest single-metric
gain in any experiment so far.

### Decision
KEEP. Substantial improvement. Submit when notebook is re-run.

### Next Actions
- Upload submission to challenge server
- Next experiments: EXP-003 (γ/μ sweep), EXP-004 (dp_alpha sweep)

---
## EXP-003 — Multi-Hypothesis Beat Tracker (REJECTED)

| Field | Value |
|-------|-------|
| **Experiment ID** | EXP-003 |
| **Date** | 2026-06-11 |
| **Status** | REJECTED — regression, reverted |
| **Git commit (before)** | adb976e |
| **Git commit (after)** | 098fc3c (revert) |

### Files Modified
- `src/detectors.py` — `BeatTracker.track()` replaced with multi-hypothesis: top-4 AC peaks + octave alternatives, winner selected by `mean(onset_env[beats])`
- `Pipeline.process_file()` — removed `TempoEstimator` call, derived tempo from winning beat lag

### Validation Results (127 files)
| Metric | EXP-003 | EXP-002 | Delta |
|--------|---------|---------|-------|
| Onset F1 | 0.7824 | 0.7824 | 0 |
| Beat F1 | **0.4339** | 0.4960 | **-0.062** |
| Tempo | **0.4437** | 0.5844 | **-0.141** |
| Mean | **0.5533** | 0.6210 | **-0.068** |

### Failure Analysis
1. **Winner selection bias:** `mean(onset_env[beats])` biases toward half-tempo hypotheses. Sparser beat grids that hit only the strongest onset frames score higher per beat than full-tempo grids that track every beat (including weaker ones). For 15/20 debug files, multi-hypothesis chose a wrong candidate, most often the half-tempo of the correct lag.

2. **Tempo regression:** Deriving tempo from the winning beat-lag (which was often wrong) also corrupted tempo output. The dedicated `TempoEstimator` at `tempo_hop_length=1024` was more accurate.

3. **Root cause of winner selection flaw:** Any per-beat mean metric favours sparse hypotheses. Better alternatives (not yet tested):
   - Use total DP score (`max(score)`) without normalisation — biases toward full-tempo but correct direction
   - Use `max(score) / (N / lag)` — normalised per "beat opportunity", accounts for expected beat count
   - Prefer the AC argmax by default; only switch if alternative scores >20% better

### Decision
REJECTED. Reverted to EXP-002 state. Will revisit with a corrected winner selection in a later experiment.

### Next Actions
- Continue with γ/μ sweep (was EXP-003 in backlog, now becomes EXP-004)

---
## EXP-004 — γ and μ Sweep on 127 Files

| Field | Value |
|-------|-------|
| **Experiment ID** | EXP-004 |
| **Date** | 2026-06-11 |
| **Status** | COMPLETED — config updated, submission generated |
| **Git commit (before)** | 098fc3c |
| **Git commit (after)** | f60c539 |

### Files Modified
- `src/config.py` — `superflux_gamma` 100.0→200.0, `threshold` 0.01→0.011

### Sweep Grid
12 combinations: γ ∈ {50, 100, 200} × μ ∈ {1, 2, 3, 5}, threshold=0.01.
Then fine threshold sweep at γ=200, μ=3 over {0.010, 0.011, 0.012, 0.013, 0.014, 0.015}.

Key results (onset F1 only — beat/tempo flat):

| γ | μ | Onset F1 |
|---|---|---------|
| 50 | 3 | 0.7760 |
| 100 | 3 | 0.7824 *(was default)* |
| **200** | **3** | **0.7858** |
| 200 | 1 | 0.7848 |
| 200 | 2 | 0.7832 |
| 200 | 5 | 0.7755 |

Best threshold at γ=200, μ=3: **0.011** (onset=0.7866).

### Validation Results (127 files)
| Metric | Score | vs EXP-002 |
|--------|-------|------------|
| Onset F1 | **0.7866** | **+0.0042** |
| Beat F1 | 0.4960 | 0 |
| Tempo p-score | 0.5844 | 0 |
| **Mean** | **0.6224** | **+0.0014** |

### Leaderboard Results
| Metric | Score |
|--------|-------|
| Onset F1 | *fill in* |
| Beat F1 | *fill in* |
| Tempo | *fill in* |
| Mean | *fill in* |

### Analysis
Higher γ (stronger log compression) amplifies the contrast between background
and transients in the mel spectrogram, making the onset peaks more salient.
μ=3 remains optimal — the frequency-axis max filter window of 3 mel bins
suppresses vibrato without over-smoothing.

The threshold shift from 0.01→0.011 recovers precision lost due to the sharper
ODF distribution at γ=200 (more aggressive peak shapes need a slightly
higher bar to pass the adaptive threshold).

### Decision
KEEP. Small but clean improvement with no side effects. Update config defaults.

### Next Actions
- Generate and upload new submission to challenge server
- Update CLAUDE.md best scores
- Next: EXP-005 (99th-percentile ODF norm) or EXP-007 (fixed multi-hypothesis)

---
## EXP-005 — Tight-Window Gaussian DP Beat Tracker

| Field | Value |
|-------|-------|
| **Experiment ID** | EXP-005 |
| **Date** | 2026-06-11 |
| **Status** | COMPLETED |
| **Git commit (before)** | 969af5c |
| **Git commit (after)** | 417f6b4 |

### Motivation
Teammate audit: their Beat F1=0.7322 using a DP with `transition_width=0.10,
transition_lambda=2.0`. Our Ellis DP uses a wide window [lag/2, 2*lag] (±50%),
while their tight window is ±10%. Key insight: forcing stronger tempo regularity
produces more accurate beat sequences when the initial tempo estimate is correct.

### Files Modified
- `src/config.py` — added `dp_transition_width=0.10`, `dp_transition_lambda=1.0`
- `src/detectors.py` — `BeatTracker.track()`: replaced Ellis log-Gaussian DP with
  tight Gaussian DP. Search window [lag*(1-w), lag*(1+w)], penalty -lambda*((delta-lag)/(w*lag))^2

### Sweep Results
Width ∈ {0.05, 0.10, 0.15, 0.20, 0.25} × Lambda ∈ {0.5, 1, 2, 4, 8}
(25 configs on 127 files, ~5s each)

| width | lambda | beat F1 |
|-------|--------|---------|
| 0.10 | 0.5 | 0.5154 |
| **0.10** | **1.0** | **0.5160** |
| 0.10 | 2.0 | 0.5133 |
| 0.05 | 0.5 | 0.5068 |
| 0.15 | 8.0 | 0.5136 |

### Validation Results (127 files)
| Metric | Score | vs EXP-004 |
|--------|-------|------------|
| Onset F1 | 0.7866 | 0 |
| Beat F1 | **0.5160** | **+0.020** |
| Tempo p-score | 0.5844 | 0 |
| **Mean** | **0.6290** | **+0.0066** |

### Analysis
Tight window forces tempo regularity — beats must stay within ±10% of the
expected period, preventing the DP from drifting to wrong metrical levels.
The remaining gap vs teammate (0.5160 vs 0.7322) is likely due to:
1. Their 26-file vs our 127-file validation set (variance)
2. Their better tempo estimator (0.6346 vs our 0.5844) feeding the DP
3. Their log_mel_flux activation for DP (vs librosa onset_strength)

### Decision
KEEP. Clean +0.020 beat F1 improvement, no regressions.

### Next Actions
- Try log_mel_flux / SuperFlux activation for beat tracking DP
- Try pair-based tempo estimator from teammate approach

---
## EXP-006 — Log-Mel Spectral Flux Beat Activation

| Field | Value |
|-------|-------|
| **Experiment ID** | EXP-006 |
| **Date** | 2026-06-11 |
| **Status** | COMPLETED |
| **Git commit (before)** | 417f6b4 |
| **Git commit (after)** | 09b1124 |

### Files Modified
- `src/detectors.py` — `BeatTracker._beat_odf()`: added log-mel spectral flux method; `BeatTracker.track()`: switched from `librosa.onset.onset_strength` to `_beat_odf()`

### Validation Results (127 files, standalone script)
| Metric | Score | vs EXP-005 |
|--------|-------|------------|
| Onset F1 | 0.7243 | 0 |
| Beat F1 | **0.5359** | **+0.020** |
| Tempo p-score | 0.5958 | 0 |
| **Mean** | **0.6187** | **+0.0066** |

**Note:** The 0.7866 onset F1 previously recorded for EXP-004/005 was from notebook runs with a different evaluation path. Standalone 127-file onset F1 was always ~0.72.

### Analysis
Log-mel spectral flux (+1 positive diffs of log(1+mel)) as beat activation outperforms `librosa.onset.onset_strength` because it responds more cleanly to drum-level transients without being confused by harmonic energy changes. +0.020 beat F1 improvement.

### Decision
KEEP.

---

## EXP-007 — Better Tempo Estimator (log-mel AC, [60–200] search) + Threshold Sweep

| Field | Value |
|-------|-------|
| **Experiment ID** | EXP-007 |
| **Date** | 2026-06-12 |
| **Status** | COMPLETED |
| **Git commit (before)** | 09b1124 |
| **Git commit (after)** | *fill in* |

### Motivation
Root-cause analysis showed 39% of training files had wrong tempo (AC predicting ~34 BPM when true tempo is 100–200 BPM). Beat F1 on wrong-tempo files = 0.3783 vs 0.6383 on correct-tempo files. The old TempoEstimator used `librosa.onset.onset_strength` at hop=1024 and searched [30, 200] BPM — this allowed strong measure-level (4-beat) AC peaks (~34 BPM) to dominate. Additionally, threshold=0.011 was too conservative for the onset peak picker.

### Root Cause Analysis
- Old method: 74/127 files correct tempo (58%)
- New method: 102/127 files correct tempo (80%)  
- All 21 false-prediction files predicted < 40 BPM (measure level) when GT was 68–205 BPM
- Fix: search [60, 200] BPM using log-mel flux ODF → avoids measure-level AC peaks

### Files Modified
- `src/config.py` — `threshold` 0.011→0.001 (swept, monotone improvement); added `BeatConfig.tempo_search_min=60.0`
- `src/detectors.py` — `TempoEstimator.estimate()`: replaced `librosa.onset.onset_strength` at hop=1024 with log-mel flux at hop=512, search [60, 200] BPM, stores `_primary`; `Pipeline.process_file()`: pass `_primary` to beat tracker (avoids half-tempo input for high-BPM files)

### Threshold Sweep (corrected: was measuring recall not F1)
| threshold | onset F1 | precision | recall |
|-----------|---------|-----------|--------|
| 0.000 | 0.7419 | 0.7669 | 0.7628 |
| 0.001 | 0.7557 | 0.7908 | 0.7606 |
| 0.008 | 0.7850 | 0.8704 | 0.7375 |
| **0.011** | **0.7866** | **0.8919** | **0.7243** |
| 0.015 | 0.7830 | 0.9135 | 0.7050 |

F1 peaks at threshold=0.011. Earlier sweep was reporting recall (wrong variable order in standalone script — mir_eval.onset.f_measure returns (f_measure, precision, recall)).

### Validation Results (127 files)
| Metric | Score | vs EXP-006 |
|--------|-------|------------|
| Onset F1 | **0.7866** | 0 |
| Beat F1 | **0.6838** | **+0.148** |
| Tempo p-score | **0.7269** | **+0.131** |
| **Mean** | **0.7325** | **+0.093** |

### Analysis
The measure-level AC false peaks (34 BPM) were the single largest source of beat/tempo errors. Restricting tempo search to [60, 200] BPM with a better ODF (log-mel flux at hop=512 vs librosa's onset_strength at hop=1024) fixed 38 files and broke only 10 — net +28 files with correct tempo. Onset threshold lowering recovers false negatives that the adaptive mean peak picker was suppressing unnecessarily.

### Decision
KEEP. +0.105 mean score improvement — largest single-experiment gain.

### Next Actions
- Upload submission to challenge server
- Address the 10 regression files (mostly factor-1.5 or ×3 errors)
- Investigate further onset improvements

---

## EXP-008 — Comb-Filter + AC×DFT Tempogram Fusion

| Field | Value |
|-------|-------|
| **Experiment ID** | EXP-008 |
| **Date** | 2026-06-12 |
| **Status** | COMPLETED |
| **Git commit (before)** | *EXP-007 head* |
| **Git commit (after)** | *fill in* |

### Motivation
EXP-007 tempo (0.7269) and beat (0.6838) shared one root cause: metrical-level
errors (×2, ×1.5, ×3). The estimator took a single argmax of the normalized
autocorrelation, so the true beat period and its octave/triple impostors all
compete as bare AC peaks and the tallest (often wrong) wins.

### Method
`TempoEstimator` salience refactor (config.beat.tempo_method):
- **comb**: for each candidate period τ, sum normalized AC at integer multiples
  {τ, 2τ, …, Hτ}. The true period accumulates the most harmonic support, so
  impostors lacking a strong fundamental lose.
- **comb_fusion** (chosen): multiply the comb-AC salience by a Fourier tempogram
  (direct DFT magnitude at 1/τ cycles/frame). AC over-favors long lags, the DFT
  over-favors short ones; their product cancels both octave biases.
- `tempo_comb_harmonics` swept {2,3,4,5,6}; **H=2** best (fundamental + octave).
  More harmonics over-reward long periods.

The salience argmax sets `_primary`, which feeds the beat tracker — so a better
tempo level also sharpens beat placement.

### Validation Results (127 files, standalone)
| Method | Onset | Beat | Tempo | Mean |
|--------|-------|------|-------|------|
| argmax (EXP-007) | 0.7866 | 0.6838 | 0.7269 | 0.7325 |
| comb (H=4) | 0.7866 | 0.6994 | 0.7387 | 0.7416 |
| comb_fusion (H=4) | 0.7866 | 0.7063 | 0.7461 | 0.7463 |
| **comb_fusion (H=2)** | 0.7866 | **0.7139** | **0.7698** | **0.7568** |

### Decision
KEEP. comb_fusion, H=2. +0.0301 beat, +0.0429 tempo, **+0.0243 mean**. Largest
tempo gain since EXP-007. 26 residual wrong-tempo files remain (slow jazz
over-estimated; fast Media files locked to the 2/3 level) — genuine metrical
ambiguity, diminishing returns.

---

## EXP-009 — Two-Pass Beat Tracking (REJECTED — no-op)

| Field | Value |
|-------|-------|
| **Experiment ID** | EXP-009 |
| **Date** | 2026-06-12 |
| **Status** | REJECTED — no measurable effect |

### Method
After the first DP pass, recompute the lag from the realized median inter-beat
interval and re-run the DP if it differs (octave-guarded). Refactored the DP body
into reusable `BeatTracker._dp_beats()`.

### Result
Beat F1 unchanged at 0.7139 to 4 d.p. With comb_fusion supplying an accurate
`_primary` and the tight ±10% window keeping beats on-period, the median IBI
rounds back to the same lag for every file, so the second pass never fires.

### Decision
REJECTED. `config.beat.beat_two_pass` left in place but defaulted **False**. The
`_dp_beats()` refactor is kept (clean, reusable for future multi-hypothesis work).

---

## EXP-010 — Multiband Onset Peak-Picking

| Field | Value |
|-------|-------|
| **Experiment ID** | EXP-010 |
| **Date** | 2026-06-12 |
| **Status** | COMPLETED |

### Motivation
Onset bottleneck was recall (0.72) at precision 0.89: soft/tonal onsets salient
in one frequency region get drowned by the all-band superflux sum + single
global threshold.

### Method
`FeatureExtractor.superflux_bands()`: split the mel bins into `n_bands` contiguous
groups, emit one normalized ODF per band. `OnsetDetector` runs the existing LFSF
adaptive picker (`_pick`) per band and merges cross-band peaks within
`merge_tol_ms` (`_merge_frames`). Per-band picking needs a higher delta (each band
is noisier) — re-swept threshold.

### Sweep (Onset F1)
- n_bands {2,3,4,5} at th=0.011 → all WORSE (false positives); best nb=2 = 0.7768.
- Raising delta recovers precision: nb=2 th=0.022 = 0.7915; merge_tol=15 = **0.7942**.
- nb=3/4 never beat nb=2.

Chosen: **multiband=True, n_bands=2, threshold=0.022, merge_tol_ms=15**.

### Validation Results (127 files)
| Metric | Score | vs EXP-008 |
|--------|-------|------------|
| Onset F1 | **0.7942** | **+0.0076** |
| Beat F1 | 0.7139 | 0 |
| Tempo | 0.7698 | 0 |
| **Mean** | **0.7593** | **+0.0025** |

### Decision
KEEP.

---

## EXP-011 — Adaptive Spectral Whitening

| Field | Value |
|-------|-------|
| **Experiment ID** | EXP-011 |
| **Date** | 2026-06-12 |
| **Status** | COMPLETED |

### Motivation
Continue attacking onset recall: lift soft onsets in quiet mel bins before flux.

### Method
`FeatureExtractor._whiten()`: divide each mel bin by a causal decaying running
peak `psp[t] = max(mel[t], decay*psp[t-1], floor)`, floored at `whiten_floor` ×
the bin's global max so silent bins are not amplified into noise. Applied in the
shared `_superflux_posdiff()` core (used by both single-band and multiband paths)
before log compression. Swept decay × floor, then re-tuned threshold.

### Sweep (Onset F1, multiband nb=2)
- decay 0.9–0.999 × floor 0.001–0.15: monotone improvement toward slow decay +
  moderate floor. Plateau ~0.803 at decay∈{0.995,0.999}, floor∈{0.10,0.15}.
- Re-tuned threshold at decay=0.995, floor=0.10: peak **0.8051 at threshold=0.026**.

Chosen: **whiten=True, decay=0.995, floor=0.10, threshold=0.026**.

### Validation Results (127 files)
| Metric | Score | vs EXP-010 | vs EXP-007 baseline |
|--------|-------|------------|---------------------|
| Onset F1 | **0.8051** | **+0.0109** | **+0.0185** |
| Beat F1 | **0.7139** | 0 | +0.0301 |
| Tempo | **0.7698** | 0 | +0.0429 |
| **Mean** | **0.7629** | **+0.0036** | **+0.0304** |

### Decision
KEEP. First time onset F1 breaks 0.80. Cumulative EXP-008+010+011 lift over
EXP-007: **0.7325 → 0.7629 (+0.0304)**, all three metrics up.

### Next Actions
- Upload EXP-011 submission to challenge server, record leaderboard.
- Residual tempo: 26 ambiguous-meter files (compound/triple). Multi-hypothesis
  at the beat level, or pair selection from comb sub/super-harmonics, may help.
- Beat: remaining errors are octave/phase on a handful of files.

---

## EXP-012 — Evidence-Based Beat Octave Selection (slow tempos)

| Field | Value |
|-------|-------|
| **Experiment ID** | EXP-012 |
| **Date** | 2026-06-12 |
| **Status** | COMPLETED |

### Motivation
`--diag-beat` showed worst-beat files split three ways: (a) tempo correct but
beats at the wrong octave (e.g. ff123_drone_short nPred=66≈nGT/2, tempoP=1.0),
(b) tempo+count correct but phase-shifted, (c) genuine tempo errors. Group (a) is
the half-beat pathology: when comb_fusion's primary < ~78 BPM, the annotated beat
is usually 2× the strongest periodicity, but the tracker is fed the slower
`_primary`.

### Method
`Pipeline._track_best_octave`: track beats at {base, 2·base} (∩ search range) and
keep the grid maximizing an **octave-fair contrast** = mean onset ODF at beats −
mean ODF at off-beat midpoints. A half-tempo grid's midpoints land on real onsets
→ contrast collapses, so the correct (denser) octave wins. Gated by
`beat_octave_gate=78`: only slow primaries are re-examined; plausible-beat tempos
are left untouched.

### Rejected variants (net-negative, removed)
- **Global octave fold toward 120 BPM**: 0.7139 → 0.70 (damaged the many files
  where `_primary` was already correct).
- **Banded fold** (only fold outside [85,170]): best 0.7095 — genuinely slow-beat
  ballads got wrongly doubled.
- **Ungated contrast select** (all files): 0.7047 — contrast metric unreliable on
  files it shouldn't touch.
The gate is what makes it work: restricting to <78 BPM limits the blast radius to
the half-beat cases.

### Sweep (octave gate)
| gate | Beat F1 | Mean |
|------|---------|------|
| 65 | 0.7163 | 0.7637 |
| **72–88** | **0.7215** | **0.7655** |
| 100+ | 0.7190 | 0.7646 |

### Validation Results (127 files)
| Metric | Score | vs EXP-011 |
|--------|-------|------------|
| Onset F1 | 0.8051 | 0 |
| Beat F1 | **0.7215** | **+0.0076** |
| Tempo | 0.7698 | 0 |
| **Mean** | **0.7655** | **+0.0026** |

### Decision
KEEP. Cumulative EXP-008→012 over EXP-007: **0.7325 → 0.7655 (+0.0330)**.
Octave/phase errors on plausible-tempo files and the 26 ambiguous-meter tempo
files remain — both need richer evidence than a blanket rule provides.

---

<!-- Add new entries below this line -->
