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
<!-- Add new entries below this line -->
