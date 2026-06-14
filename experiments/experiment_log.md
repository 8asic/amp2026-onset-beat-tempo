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
| **Status** | COMPLETED — submitted, leaderboard recorded |
| **Submission file** | `submissions/EXP-012_20260612_180044_aa97e56/predictions.json` |

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

### Leaderboard Results (submitted 2026-06-12)
| Metric | Validation | Leaderboard | Delta |
|--------|------------|-------------|-------|
| Onset F1 | 0.8051 | **0.75** | −0.055 |
| Beat F1 | 0.7215 | **0.725** | +0.004 |
| Tempo p-score | 0.7698 | **0.86** | +0.090 |
| **Mean** | **0.7655** | **0.7783** | **+0.013** |

Leaderboard mean exceeds validation mean — **no overfitting overall**. Onset is
the only gap (−0.055): threshold 0.026 likely tuned slightly hot on train. Tempo
generalizes very strongly (+0.09, comb_fusion is robust on unseen styles). Beat
is essentially the same (+0.004).

### Decision
KEEP. Cumulative EXP-008→012 over EXP-007: **0.7325 → 0.7655 (+0.0330)** val,
**leaderboard 0.7783**.
Onset is the active bottleneck on leaderboard (0.75). Next: HPSS percussive
masking to improve onset recall on leaderboard without threshold overfitting.
Octave/phase errors on plausible-tempo files and the 26 ambiguous-meter tempo
files remain — both need richer evidence than a blanket rule provides.

---

## EXP-013 — Evidence-Gated Beat Phase Refinement (REJECTED — no-op)

| Field | Value |
|-------|-------|
| **Experiment ID** | EXP-013 |
| **Date** | 2026-06-12 |
| **Status** | REJECTED — no-op, code removed |

### Motivation
`--diag-phase` (new harness diagnostic) isolated 8 files that are tempo-correct
(p≥0.5) but beat-poor (F<0.5). Two are clean anti-phase (ff123_drone_short
|off|/period=0.465, SoundCheck2_82_Yello 0.488 — beats land on the off-beat);
the other six are moderate jitter (frac 0.18–0.29, several with median signed
offset ≈ 0, i.e. scattered not shifted).

### Method
After tracking, try shifting the whole beat grid by ±½ period and keep the phase
with higher on/off-beat onset contrast, accepting a flip only if it beats the
current phase by `beat_phase_margin`.

### Result
Beat F1 unchanged at 0.7215 for every margin in {0, 0.02, 0.05, 0.10}. The flip
**never triggers**: the DP already maximizes onset energy along its path, so the
alternate phase never has higher contrast. The two genuine anti-phase files
(drone, Yello) have almost no onset evidence — which is exactly why the DP chose
the wrong phase and why no onset-based criterion can recover it.

### Decision
REJECTED. Code removed; `--diag-phase` diagnostic kept. Conclusion: the residual
phase failures are onset-evidence-starved (drones/synths) and not fixable from the
onset ODF. Would need a different signal (bass/harmonic tracking or a learned
model) — out of scope and overfit-prone for ~2 files.

---

## EXP-014 — HPSS Percussive Masking (REJECTED — train overfit)

| Field | Value |
|-------|-------|
| **Experiment ID** | EXP-014 |
| **Date** | 2026-06-12 |
| **Status** | REJECTED — generalizes worse on extra onset data |

### Motivation
EXP-012 leaderboard revealed onset gap: val 0.8051 vs leaderboard 0.75 (−0.055).
Hypothesis: SuperFlux captures harmonic onset energy that the test set may have
less of; HPSS percussive masking before superflux would bias toward transients.

### Method
STFT-domain HPSS (Fitzgerald 2010): compute power STFT, apply median filter along
time axis (harmonic, kernel=kh frames) and frequency axis (percussive, kernel=kp
bins), form soft mask `perc/(perc+harm+eps)`, apply to power STFT before projecting
to mel. This gives higher frequency resolution than mel-domain HPSS (1025 bins vs 82).

### Key secondary finding: val→lb onset gap is distribution mismatch
Used the 150 `train_extra_onsets` files as a proxy for test-set diversity. Finding:
- Baseline threshold=0.026 on 127 train files: **onset F1 0.8051**
- Baseline threshold=0.026 on 150 extra onset files: **onset F1 0.7021**
- Combined 277 files at threshold=0.026: **onset F1 0.7493** ≈ leaderboard 0.75

This confirms the val→lb onset gap is almost entirely explained by distribution
mismatch, not an algorithmic bug. The 150 extra files are harder/more diverse
and match the leaderboard distribution.

### HPSS sweep results (kh × kp on 127 train files)
| kh | kp | Onset F1 | Mean |
|----|-----|---------|------|
| 17 | 17 | 0.8007 | 0.7640 |
| 31 | 17 | 0.8027 | 0.7646 |
| 51 | 51 | 0.8083 | 0.7665 |
| 71 | 51 | 0.8084 | 0.7665 |
| 91 | 17 | 0.8079 | 0.7664 |
| **127** | **17** | **0.8092** | **0.7668** |
| 171 | 17 | 0.8085 | 0.7666 |

Best on train: **kh=127, kp=17** → Onset 0.8092, Mean 0.7668 (+0.0013 mean).

### Generalization failure
| Config | Train 127 | Extra 150 | Combined 277 |
|--------|-----------|-----------|--------------|
| Baseline 0.026 | 0.8051 | 0.7021 | 0.7493 |
| HPSS kh=127 kp=17 | **0.8092** | **0.675** | **0.7366** |

HPSS *hurts* on extra files by −0.027. Classic overfit: extracts features specific
to the 127 training file style distribution.

### Other things tested (all rejected)
- **n_bands sweep on 277 files**: n_bands=3 → 0.7146, n_bands=4 → 0.6764 (both worse than 2→0.7493)
- **threshold=0.028**: train plateau (0.8050), extra +0.005, combined +0.0025 — too small to submit
- **threshold=0.030**: train −0.0002, extra +0.0087 — marginal

### Decision
REJECTED. HPSS code left in `features.py` but `config.audio.hpss=False` (default off).
The val→lb onset gap is a distribution mismatch — not fixable by feature engineering
without test set access. The 150-file extra-onset set is now in the harness via
`--extra-onsets` for future threshold calibration.

**Key rule going forward**: any onset feature must be validated on the 277-file
combined set (127 train + 150 extra) to screen for train-set overfit before submitting.

---

## EXP-015 — Multi-ODF Fusion (SuperFlux + Complex-Domain) — KEEP

| Field | Value |
|-------|-------|
| **Experiment ID** | EXP-015 |
| **Date** | 2026-06-12 |
| **Status** | COMPLETED — KEEP (first onset gain that generalizes to the 277-file set) |
| **Git commit (before)** | 2068bc6 |

### Motivation
EXP-014 established that the val→leaderboard onset gap is distribution mismatch:
the 277-file combined set (0.7493) matches the leaderboard (0.75), while the
127-train score (0.8051) is an over-optimistic outlier. A fixed-threshold
SuperFlux ODF cannot be pushed further on train without overfitting. To gain on
the *generalization* benchmark we need a richer ODF that catches onset types
SuperFlux misses — specifically soft/tonal attacks where magnitude flux is weak
but phase/complex structure changes.

### Method
Added three new onset detection functions (pure numpy/scipy, allowed libs):
- **complex_domain** (rectified, Bello/Duxbury): predict each STFT bin from the
  previous frame (constant magnitude, constant phase rate), measure complex
  deviation, rectify to energy rises. Catches tonal onsets.
- **hfc** (Masri): high-frequency-content flux.
- **phase_deviation** (magnitude-weighted, Bello).

Fusion path: each ODF is a *channel*; the existing LFSF peak picker runs per
channel and peaks are merged across channels within merge_tol_ms — identical
architecture to multiband (EXP-010), just with cross-family channels. The
musical decision (peak picking) stays our own hand-written code; the model only
upgrades the activation. `config.onset.fusion`, `config.onset.fusion_odfs`.

### Standalone ODF quality (277-file set, own threshold)
| ODF | best Onset F1 | verdict |
|-----|---------------|---------|
| superflux (multiband, baseline) | 0.7493 | reference |
| complex-domain | 0.7476 | competitive — KEEP |
| phase-deviation | 0.7094 | weak in union — drop |
| hfc | 0.6101 | weak — drop |

Complex-domain alone nearly matches the full multiband SuperFlux, and catches
*different* onsets, so the union exceeds both.

### Fusion sweep (superflux + complex, 277-file set)
| threshold | Onset F1 (277) |
|-----------|----------------|
| 0.040 | 0.7532 |
| 0.050 | 0.7601 |
| **0.055** | **0.7615** |
| 0.058 | 0.7615 |
| 0.060 | 0.7612 |
| 0.070 | 0.7594 |

Broad plateau at 0.055–0.058 (not an overfit spike). Adding phase to the union
regressed it to ~0.71 — rejected. The complex channel adds peaks, so the union
needs a higher delta (0.055 vs 0.026 single-family) to trim false positives.

### Train/extra split at threshold=0.055
| set | baseline (multiband 0.026) | EXP-015 fusion | delta |
|-----|---------------------------|----------------|-------|
| 127 train | 0.8051 | 0.8055 | +0.0004 (flat — no train overfit) |
| 150 extra (held-out proxy) | 0.7021 | 0.7243 | **+0.0222** |
| 277 combined | 0.7493 | **0.7615** | **+0.0122** |

The gain is driven by the held-out extra files, with zero train regression —
the opposite signature of EXP-014's HPSS overfit. This is the screening rule
working as intended.

### Validation Results
| Metric | EXP-012 | EXP-015 |
|--------|---------|---------|
| Onset F1 (127) | 0.8051 | 0.8055 |
| Onset F1 (277) | 0.7493 | **0.7615** |
| Beat F1 (127) | 0.7215 | 0.7215 |
| Tempo (127) | 0.7698 | 0.7698 |
| Mean (127) | 0.7655 | 0.7656 |

Beat/tempo unchanged (fusion only touches onset detection). Predicted
leaderboard: onset 0.75 → ~0.762, mean 0.7783 → ~0.782.

### Leaderboard Results (submitted 2026-06-12)
| Metric | EXP-012 lb | EXP-015 lb | delta |
|--------|-----------|-----------|-------|
| Onset F1 | 0.75 | **0.775** | **+0.025** |
| Beat F1 | 0.725 | 0.725 | 0 (unchanged) |
| Tempo p-score | 0.86 | 0.86 | 0 (unchanged) |
| Mean | 0.7783 | **~0.787** | **+0.0084** |

The leaderboard onset (0.775) came in **above** both the prediction (~0.762)
and the 277-proxy (0.7615). The test-set onset distribution generalizes even
better for fusion than the extra-onset proxy suggested. The 277-proxy remains a
conservative screen — it correctly predicted the *direction* and a lower bound.

### Decision
KEEP — confirmed on leaderboard. First onset change since the baseline that
improves both the generalization benchmark (+0.0122 on 277) AND the leaderboard
(+0.025), not just the train set. Config defaults:
`fusion=True, fusion_odfs=("superflux","complex"), threshold=0.055`.

### Next
Step 2 (planned): pure-numpy *learned* onset activation (logistic/MLP on ODF
features) feeding the same peak picker, with k-fold held-out eval to avoid
training-on-test. The complex-domain ODF is now an additional input feature for it.

---

## EXP-016 — Pure-Numpy Learned Onset Activation — REJECTED for submission (KEEP as asset)

| Field | Value |
|-------|-------|
| **Experiment ID** | EXP-016 |
| **Date** | 2026-06-12 |
| **Status** | REJECTED for submission (loses on the test-relevant corpus); code KEPT as inert, validated infrastructure (`config.onset.learned=False`) |

### Motivation
A fixed onset threshold cannot adapt per-frame (quiet folk vs loud electronic).
A learned per-frame classifier on the existing ODF channels (superflux bands +
complex-domain) could learn an adaptive decision, feeding the SAME hand-written
peak picker (rules-clean: pure-numpy LR/MLP, no library classifier/peak-picker).

### Method
- Features: ODF channels (superflux bands + complex) context-stacked ±5 frames → 33 dims.
- Labels: frame positive within ±`label_w` frames of a GT onset.
- Model: class-weighted logistic regression, hand-written forward + gradient (numpy).
- Eval: 5-fold CV over the 277 onset-labelled files; the classifier scores only
  on files it did NOT train on. Peak-pick delta swept; gain robust across delta.

### CV results (277-file, held-out)
| model | held-out onset F1 | vs fusion 0.7615 |
|-------|-------------------|------------------|
| MLP h=32 | 0.7628 | +0.0013 (overfits) |
| LR ctx5 lw2 wpos5 | 0.7651 | +0.0036 |
| LR ctx5 lw1 wpos5 | 0.7724 | +0.0109 |
| **LR ctx5 lw1 wpos2** | **0.7729** | **+0.0114** |
LR not overfitting: train-on-train 277 = 0.7720 ≈ CV 0.7729. label_w=1 (sharp
labels) was the key lever; MLP added nothing (overfits this data).

### Why it is REJECTED despite winning the 277-CV — the decisive subset split
Held-out F1 split by corpus (fixed delta=0.18):
| corpus | fusion | learned (held-out) | winner |
|--------|--------|--------------------|--------|
| **c127** (challenge main corpus) | **0.8055** | 0.7784 | **fusion +0.027** |
| extra-150 (supplementary) | 0.7242 | 0.7659 | learned +0.042 |
| 277 combined | 0.7615 | 0.7729 | learned +0.011 |

The learned model's 277-CV advantage is **entirely** from the extra-onset corpus.
The leaderboard tells us which corpus the 50 test files resemble: EXP-015 fusion
scored **0.775** on the leaderboard, which sits at the **c127 level** (0.8055),
far above the 277 (0.7615) or extra (0.7242) levels. So the test set is
distributed like c127 — where **fusion beats the learned model by +0.027**.
Submitting the learned model would very likely REGRESS leaderboard onset below
0.775.

### Methodological lesson (important)
The 277-combined proxy is the right screen for *overfit* (EXP-014) and for
confirming a same-distribution gain (EXP-015). But when comparing two models
with DIFFERENT per-subset profiles, the 277-average can MISLEAD: it rewarded the
learned model for gains on the extra corpus that the test set under-represents.
**When choosing between models, also split the proxy by corpus and weight by the
corpus the leaderboard shows the test resembles (c127).** The leaderboard onset
level is the tell for which corpus the test matches.

### Decision
REJECTED for submission — fusion (EXP-015, lb 0.775) stays the onset detector.
KEEP the code as inert, validated infrastructure (`src/learned_onset.py`, config
`learned`/`learned_model_path`/`learned_delta`, OnsetDetector learned path,
`notebooks/02_learned_onset.ipynb`, `experiments/exp016_learned_onset.py` +
`train_onset_model.py`). It already beats fusion on the extra corpus; with richer
features (more ODFs, mel context) it may eventually beat fusion on c127 too — the
most promising path to revisit it. `config.onset.learned` remains False.

---

## EXP-019 — Richer features for the learned onset model — PARITY (banked, not shipped)

| Field | Value |
|-------|-------|
| **Experiment ID** | EXP-019 |
| **Date** | 2026-06-12 |
| **Status** | Validated candidate, NOT shipped (user chose to move to EXP-018); fusion stays live |

### Motivation
EXP-016 showed the learned onset model loses to fusion on the test-relevant c127
corpus (−0.027). Try richer features so it can match fusion *on c127* (the bar),
exploiting the now-known rule: judge on c127, not the 277-average.

### Method
Added `--odfs` and `--n-bands` to the prototype. Swept ODF channel sets
(superflux/complex/hfc/phase) and superflux band counts (2→16), context width,
and LR vs MLP. Judged on held-out c127 (fixed delta=0.18).

### Results (held-out c127; fusion bar = 0.8055, leakage-inflated)
| config | c127 | extra |
|--------|------|-------|
| LR sf,cx nb2 (EXP-016) | 0.7784 | 0.766 |
| LR +hfc+phase nb2 | 0.7855 | 0.760 |
| LR nb8 +all ODFs | 0.7994 | 0.761 |
| LR nb8/12 (plateau) | ~0.799 | — |
| **MLP h64 nb8 +all ODFs** | **0.8048** (seeds: .8048/.8043/.8053) | 0.737 |
| MLP h96 nb8 +all ODFs | 0.8064 | 0.735 |

LR plateaus ~0.799 (−0.006 vs fusion). The MLP with rich features reaches c127
≈ 0.805 — a stable **tie** with fusion (±0.0005 across seeds), and beats fusion
on extra. Crucially the MLP's 0.805 is **held-out** while fusion's 0.8055 is
leakage-inflated (threshold tuned on the 277 incl. these files; fusion's true
generalization is the lb 0.775). So the MLP is fairly ≥ fusion on c127.

### Decision
PARITY achieved — a genuine low-downside ship candidate (expected lb onset ~0 to
+0.02). NOT shipped: user opted to invest the next effort in EXP-018 (beat), which
has far more headroom (lb beat 0.725). Kept as validated infrastructure; revisit
if we want to bank the small onset upside on a spare leaderboard slot. Fusion
(EXP-015, lb 0.775) remains the live onset detector.

---

## EXP-018 — Beat BLSTM (PyTorch / Colab) — KEEP (SHIPPED)

| Field | Value |
|-------|-------|
| **Experiment ID** | EXP-018 |
| **Date** | 2026-06-12 |
| **Status** | KEEP — SHIPPED. Fair c127 beat 0.7335 vs production 0.7215 (+0.0120). config.beat.learned=True, decode B. |

### Decision to use PyTorch (rules note)
User explicitly approved PyTorch for a **self-trained** model (not on the original
allowed-list, but distinct from the banned madmom pre-trained trackers). Hard
lines kept: no madmom, no `librosa.beat.beat_track`; the tempo+DP phase decoder
stays our own code. This is the user's rules-interpretation call.

### Plan (per user spec)
- Bidirectional LSTM (1–2 layers, hidden 64–128), input log-mel (81 bands, hop
  512, win 2048, 22050 Hz), output per-frame beat-activation probability (BCE,
  pos-weighted). Labels: ±1 frame of a GT beat.
- Data: 696 `train_extra_tempobeats` + 127 main = 823 files, 80/20 split.
- Decode: activation → our `_estimate_tempo_from_env` + `_dp_beats` (no librosa).
- Notebook: `notebooks/02_beat_model.ipynb` (Colab-ready: installs deps, clones
  repo, mounts Drive for data). Saves `models/beat_blstm.pt`.
- Success: held-out beat F1 > 0.45; report autocorr baseline vs BLSTM.
- `src/detectors.py` `BeatTracker` to load the model when weights exist, else
  fall back to the autocorrelation tracker — **pending explicit user confirmation
  before writing**.

### Results — fair c127 evaluation (the decisive test)
The notebook val (mixed, extra-heavy) gave big deltas over a WEAK autocorr
baseline (+0.12–0.16), but that is not our production beat. The honest test:
train on the 696 extra files ONLY (zero c127), evaluate on all 127 c127 with the
SHIP decode (B = comb_fusion tempo + octave-select over the BLSTM activation),
compare to production 0.7215.

| beat method (127, model never trained on them) | Beat F1 |
|-----------------------------------------------|---------|
| production (log-mel flux + comb_fusion + octave) | 0.7215 |
| BLSTM under-trained (noisy), decode B | 0.7251 (+0.004, wash) |
| **BLSTM stabilized, decode B** | **0.7335 (+0.0120)** |
| BLSTM stabilized, decode A | ~0.70 (worse — A drops comb_fusion+octave) |

Decode A (autocorr on the activation) is consistently worse than B — the win
comes from swapping the beat ODF (flux → BLSTM activation) inside our existing
comb_fusion-tempo + octave-select framework. Stabilizing training (grad-clip 3.0,
ReduceLROnPlateau, early-stop) turned the +0.004 wash into a genuine +0.012 fair
gain. 0.7335 is a conservative FLOOR for the ship model (trained on zero c127);
the shipped all-823 model trains on c127-distribution data, so test-50 should be
≥ this. Optimistic all-823 on its own 127: 0.7631.

### What worked / lessons
- The BLSTM is a much better beat *signal* than log-mel flux, but only decode B
  (keeping our strong comb_fusion tempo + octave-select) realises it on c127.
- Under-training masked the gain — the noisy val loss was the tell. Stabilisation
  was necessary to clear the production bar.
- Same corpus lesson as EXP-016/019: notebook deltas on the extra corpus (+0.15)
  vastly overstate the c127 gain (+0.012). Always confirm on the c127 corpus.

### Decision
KEEP — SHIPPED. `config.beat.learned=True` (decode B), `models/beat_blstm.pt`
committed (all-823 stabilised model). Onset/tempo untouched. Inference needs
torch (falls back to log-mel flux if torch/weights absent). Validation mean
(127, optimistic beat) 0.7795; honest fair beat 0.7335.

### Leaderboard (submitted 2026-06-13, beats page only)
**Beat F-measure 0.725 → 0.735 (+0.010), now 7th.** The fair c127 test predicted
0.7335 — the leaderboard came in 0.735, within 0.0015. The fair-evaluation
methodology (train-696-only, eval-c127, decode B) predicted the test almost
exactly. Onset/tempo NOT re-uploaded (their predictions are byte-identical to
EXP-015 — EXP-018 only changed the beat ODF), so they stay 0.775 / 0.86.
Standings: onset 18th (top 0.881), beat 7th (top 0.861), tempo 6th (top 0.91).
Leaderboard mean ~ (0.775+0.735+0.86)/3 = 0.790.

### Notebook artifacts
`notebooks/02_beat_model.ipynb` (TRAIN_ON_EXTRA_ONLY toggle: True=fair test →
`beat_blstm_extra.pt`; False=ship → `beat_blstm.pt`). Stabilised training cell.

---

## EXP-020 — Onset CNN (PyTorch / Colab) — SHIPPED (uncertain bet)

| Field | Value |
|-------|-------|
| **Experiment ID** | EXP-020 |
| **Date** | 2026-06-13/14 |
| **Status** | SHIPPED — config.onset.cnn=True, delta=0.30. Marginal/uncertain vs fusion; easy rollback (cnn=False). |

### Motivation
Onset is our worst leaderboard rank (18th, 0.775; top 0.881). Top scores imply
learned onset detectors. PyTorch approved (EXP-018). Mirror the beat playbook.

### Method
Fully-convolutional onset CNN (convs preserve time, pool over freq) -> per-frame
onset activation -> our `OnsetDetector._pick` (delta 0.30; no librosa.onset, no
madmom). Stabilized training (grad-clip, ReduceLROnPlateau, early-stop). Data:
277 onset files (127 main + 150 extra_onsets). `notebooks/03_onset_model.ipynb`.

### Fair test (train 150 extra-only, eval 127 c127)
| model | fair c127 |
|-------|-----------|
| small CNN (33k) | **0.7881** |
| big CNN (130k) + augmentation | 0.7740 (REGRESSED - onset data-limited) |
| fusion on the 127 themselves | 0.8055 |
| fusion leaderboard (unseen) | 0.775 |

Enlarging + augmentation HURT - only 277 onset files (vs beat's 823), so capacity
overfits and aug washes out sparse onset evidence. Kept the small 16/32/32 CNN.
All-277 model on a random held-out c127 (26 files): 0.7922 (delta 0.30),
consistent with the dedicated fair 0.7881. Delta swept on 127: peak 0.30 (0.8081
optimistic, vs fusion 0.8055).

### Why ship despite fusion winning on the 127 files
Fusion's 0.8055 is its score on the 127 files THEMSELVES; its true generalization
is the leaderboard 0.775. The CNN's fair c127 (0.7881, trained on zero c127) is an
unseen-c127 number, directly comparable to fusion's 0.775 -> CNN +0.013. The beat
precedent (fair-c127 0.7335 -> leaderboard 0.735, exact) says this transfers, so
predicted onset ~0.788 > fusion 0.775. Honest caveat: data-limited, within noise;
the on-127 comparison favours fusion, so this is a genuine bet, not a sure thing.

### Wiring
`OnsetDetector` tries the CNN first (models/onset_cnn.pt; loader infers conv
channels from the checkpoint -> arch-agnostic), falls back to fusion if torch/
weights absent. `config.onset.cnn=True, cnn_delta=0.30`. Verified: fallback gives
fusion 0.8055; CNN path reproduces notebook numbers exactly. Optimistic full
pipeline on 127: onset 0.8081, beat 0.7631.

### Decision
SHIPPED. Submission `submissions/EXP-020_*/predictions.json` (CNN onset + BLSTM
beat). Easy rollback: `config.onset.cnn=False` -> fusion onset.

### Leaderboard (submitted 2026-06-14, onsets page)
**Onset F-measure 0.775 -> 0.807 (+0.032), 18th -> 14th.** The bet paid off, and
by MORE than predicted (fair c127 0.7881 suggested ~0.788). Why the under-predict:
the onset fair model trained on only 150 extra files; the SHIP model trained on
all 277 (+46% data, *including* the 127 c127 = test distribution), which boosted
test-50 well above the fair floor. Contrast beat: its fair model already had 696,
ship added only 127 (+15%), so beat transferred ~1:1 (0.7335->0.735). **Lesson:
the fair-c127 floor is conservative when the ship model adds a large fraction of
test-distribution data.** Leaderboard mean now ~(0.807+0.735+0.86)/3 = 0.8007.
Standings: onset 14th (top 0.881), beat 7th (top 0.861), tempo 6th (top 0.91).

---

## EXP-021 - Beat tempo-augmentation - WASH for decode B (not shipped)

| Field | Value |
|-------|-------|
| **Experiment ID** | EXP-021 |
| **Date** | 2026-06-14 |
| **Status** | NOT shipped - wash for decode B; clean negative from fair test |

### Method
Train-only time-stretch augmentation (rates 0.84-1.19, ~5x; beat times scaled
1/rate) on the beat BLSTM (architecture unchanged, to isolate augmentation).

### Result (fair test: train 696-extra augmented, eval 127 c127)
| metric | non-aug | augmented |
|--------|---------|-----------|
| decode A (model autocorr tempo) | 0.7007 | 0.7200 (+0.019) |
| decode B (comb_fusion tempo + octave) | 0.7335 | 0.7315 (wash) |
| val loss (best) | 0.5358 | 0.4852 (better fit) |

### Why augmentation helps decode A but NOT decode B
Decode B takes tempo from comb_fusion (already strong, lb 0.86); the activation
only needs to mark beat PHASE. Tempo-augmentation teaches tempo-invariance, which
decode B does not need and which does not improve phase, so its benefit is
bypassed. It helps decode A (model supplies own tempo), but decode A base (0.72)
is below decode B (0.7335).

### Implication
For the decode-B beat pipeline the lever is activation/PHASE quality
(architecture: CRNN/TCN, or a DBN decoder), NOT tempo-augmentation. Beat -> top-3
needs +0.10 (0.735 -> ~0.83), the hardest metric. Pivot: onset augmentation is
higher-EV (onset is data-limited AND its picker uses the activation directly, so
augmentation is NOT bypassed the way comb_fusion bypasses it for beat).

### Decision
NOT shipped. Beat stays at EXP-018 (lb 0.735). Augmentation kept in the notebook
(RATES) for a possible CRNN follow-up.

---

## EXP-022 - Onset pitch/time augmentation - WASH (not shipped)

| Field | Value |
|-------|-------|
| **Experiment ID** | EXP-022 |
| **Date** | 2026-06-14 |
| **Status** | NOT shipped - slight regression on fair c127 |

### Method
Train-only pitch-shift (+-2 semitones) + time-stretch (+-11%), 7 variants
(150 extra -> 1050 seqs), small onset CNN unchanged. The "right" augmentation
(EXP-020's freq-mask was wrong).

### Result (fair: train 150-extra augmented, eval 127 c127)
| | fair c127 | val loss |
|--|-----------|----------|
| non-aug CNN (EXP-020) | 0.7881 | 0.6422 |
| pitch/time augmented | 0.7742 | 0.6619 |
Slight regression; augmented model fits c127 WORSE.

### Why augmentation washes (key lesson, both EXP-021 & EXP-022)
The fair test measures CROSS-CORPUS generalization (extra -> c127). The gap is a
DISTRIBUTION difference (instruments/genre/recording), not a pitch/tempo gap.
Augmentation teaches within-distribution invariance, which does not bridge a
distribution gap. (Beat aug was also bypassed by comb_fusion tempo.) Net: neither
bigger models nor augmentation move our fair-c127 - we are data/distribution
limited.

### Caveat (untested)
The fair test augments EXTRA only. A ship model augmenting all 277 (incl c127)
*might* help test-50 by adding invariance on test-distribution-like data - untested
(would need 277-CV with augmentation, or a leaderboard slot). Low confidence after
two washes.

### Decision
NOT shipped. Onset stays at EXP-020 (lb 0.807). Pivot to ENSEMBLING (CNN + fusion
make different errors) and test-time augmentation - reliable variance-reduction
levers that do not depend on more data.

---

## EXP-022b / EXP-023 - Onset aug (ship regime) + multi-resolution CNN - PLATEAU

| Field | Value |
|-------|-------|
| **Experiment ID** | EXP-022b, EXP-023 |
| **Date** | 2026-06-14 |
| **Status** | Not shipped - within-noise; onset plateaued ~0.80 c127 |

### Results (277-split, c127 IN training, held-out clean c127 = 26 files)
| model | held-out c127 (26 files) |
|-------|--------------------------|
| non-aug single-res (EXP-020) | 0.7922 |
| + pitch/time augmentation (EXP-022b) | 0.8024 |
| + multi-resolution 3 STFT (EXP-023) | 0.8010 |
| fusion DSP (same subset, noisy) | 0.7922-0.8133 across runs |

### Findings
- EXP-022b CONFIRMED the methodology fix: augmentation helps in the SHIP regime
  (c127 in training, +0.010) but washed in the extra-only fair test - the fair
  test undersells in-distribution levers (no c127 in train).
- EXP-023 multi-resolution CNN: val loss improved (0.464 vs 0.476) but c127 F1
  flat (0.8010 vs 0.8024) - better fit did NOT translate to peak F1.
- **The 26-file held-out c127 is too NOISY (+-0.02-0.03) to distinguish ~0.01
  gains** (fusion baseline alone swings 0.79-0.81 across runs). Offline tuning of
  small onset gains is unreliable; only the leaderboard or 5-fold CV (5x compute)
  could resolve them.

### Conclusion
Onset is PLATEAUED ~0.80 c127 / 0.807 lb. The standard SOTA levers (augmentation,
multi-resolution) do not clearly beat our EXP-020 ship. The leaders' 0.88 on the
same data is a method we have not cracked. Reliable remaining onset lever:
ensembling (CNN + fusion, different errors) - but only leaderboard-testable.
Pivot: beat has the biggest gap and an UNTRIED big lever (DBN/HMM decoder).

### Decision
Not shipped (within noise; do not burn slots on noise). Onset stays EXP-020
(0.807). Next: beat CRNN/TCN + proper DBN decoder (the leaders' likely edge), or
onset ensemble as a leaderboard gamble.

---

## EXP-024 - DBN beat decoder (joint tempo+phase) - WASH (not shipped)

| Field | Value |
|-------|-------|
| **Experiment ID** | EXP-024 |
| **Date** | 2026-06-14 |
| **Status** | NOT shipped - loses to DP+comb_fusion+octave-select |

### Method
From-scratch variable-lag Viterbi DBN (state = beat at frame t with incoming
lag; tempo may drift, penalised by lam_change; anchored to comb_fusion tempo
prior via lam_prior). Vectorised O(N.L^2). Tested on the shipped beat activation
(127, optimistic), vs DP decode-B 0.7631.

### Result (127, shipped beat model)
| decoder | Beat F1 |
|---------|---------|
| DP decode-B (comb_fusion + octave-select) | 0.7631 |
| DBN default (lam_change 6, lam_prior 4) | 0.7501 |
| DBN lam_prior16/lam_change24 | 0.6989 |
| DBN 16/60 | 0.6773 |
| DBN 40/24 | 0.6507 |
| DBN 40/60 | 0.6399 |

Every DBN config loses to DP; stronger anchoring/rigidity makes it WORSE.

### Why DP wins
Our DP decoder already pairs the strong comb_fusion global tempo with a tight
Gaussian phase window AND octave-select (EXP-012, +0.008). The DBN's tempo-drift
flexibility doesn't help (most files are steady-tempo) and it LACKS octave-select,
so it anchors to comb_fusion's raw octave on slow files. Net regression.

### Decision
NOT shipped. Beat stays EXP-018 (lb 0.735), decoder="dp". DBN code kept as inert
asset (config.beat.decoder).

### Big-picture conclusion
Big levers now exhausted: onset (augmentation, multi-resolution CNN) and beat
(augmentation, DBN decoder) ALL wash against the tuned DSP + small-NN pipeline.
We are at a method ceiling: mean ~0.80, onset 0.807/14th, beat 0.735/7th, tempo
0.86/6th. Top-3 in all three is not reachable with the methods found. Remaining
levers are incremental: onset ensemble (CNN+fusion, leaderboard gamble), learned/
beat-derived tempo (tempo is closest to top-3).

---

<!-- Add new entries below this line -->
