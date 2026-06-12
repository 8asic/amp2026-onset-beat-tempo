# CLAUDE.md

## Project

JKU AMP Challenge — implement onset detection, beat tracking, and tempo
estimation from scratch. Predictions are submitted to
https://challenges.cp.jku.at for scoring.

## Challenge Rules (non-negotiable)

**Allowed from libraries:**
- Audio loading and resampling (`librosa.load`, `soundfile`)
- Spectrograms, mel spectrograms, semitone spectrograms
- Generic signal processing (`scipy.signal`, `numpy`)
- Linear algebra (`numpy`, `scipy.linalg`)
- Evaluation only: `mir_eval`

**Must implement yourself — do not use library versions of:**
- Peak picking of any kind (this includes `librosa.util.peak_pick`,
  `scipy.signal.find_peaks` used as a peak picker, or any wrapper)
- Onset detectors (`librosa.onset.onset_detect`)
- Beat trackers (`librosa.beat.beat_track`, any madmom tracker)
- Tempo estimators (`librosa.beat.tempo`, `librosa.feature.rhythm.tempo`)
- Any ready-made audio feature not in the allowed list above

**When in doubt, implement it yourself.** The test is: does the library
function *make a musical decision* (what is a peak, what is a beat, what is
the tempo)? If yes, it must be your own code.

**Allowed grey area:** `numpy.argmax`, `numpy.correlate`, array slicing,
`librosa.frames_to_time` — these are arithmetic, not musical decisions.

## Repository Structure

```
src/
  config.py        — all hyperparameters (single source of truth)
  features.py      — FeatureExtractor: spectral_flux(), superflux()
  detectors.py     — OnsetDetector, BeatTracker, TempoEstimator, Pipeline
  evaluation.py    — Evaluator (wraps mir_eval, not used in submission)
  data_loader.py   — DataLoader: load_train(), load_test(), load_extra_*()
  utils.py         — file I/O helpers, save_versioned_submission()
  __init__.py

notebooks/
  01_pipeline.ipynb  — main notebook (clean, top-to-bottom reproducible)

submissions/
  <EXP-ID>_<timestamp>_<commit>/
    predictions.json   — versioned submission (never overwrite, always version)
    metadata.json      — experiment ID, git commit, val scores, notes

data/
  raw/               — zip archives (gitignored)
  processed/
    train/           — 127 annotated files (.wav + .onsets.gt .beats.gt .tempo.gt)
    test/            — 50 unannotated files (.wav only)
    train_extra_onsets/     — 150 extra onset-only files
    train_extra_tempobeats/ — 696 extra beat/tempo files

experiments/
  experiment_log.md      — completed experiments only
  experiment_backlog.md  — proposed experiments only
```

## Source Code Conventions

- Python 3.10+, type hints everywhere
- All hyperparameters live in `src/config.py` — never hardcoded elsewhere
- `config` is a singleton; import it with `from .config import config`
- The notebook may set config values explicitly in one cell (Section: Parameters)
  and nowhere else — the sweep cell must restore values after running
- No monkey-patching of class methods from notebooks

## Notebook Execution Order

The notebook must be runnable top-to-bottom with `Kernel > Restart & Run All`.
Sections in required order:

1. **Setup** — imports, project root, config
2. **Data** — extract zips (idempotent), load train/test
3. **Parameters** — set `config.*` values here only
4. **Validate** — evaluate on training set (all 127 files)
5. **Submit** — generate and save versioned submission via `save_versioned_submission()`
6. **Debug** *(optional)* — visualise individual files, clearly labelled
7. **Sweep** *(optional/exploratory)* — parameter search, restores config at end

Submission generation (Section 5) must always come **after** parameter
setting (Section 3) and validation (Section 4).

## Submission Workflow

Never write directly to `submissions/predictions.json`.
Always use `save_versioned_submission()` from `src/utils.py`:

```python
from src.utils import save_versioned_submission

pred_path = save_versioned_submission(
    predictions=submission,          # dict of {stem: {onsets, beats, tempo}}
    submissions_dir=config.paths.submissions_dir,
    experiment_id="EXP-002",
    val_scores={"onset_f1": 0.62, "beat_f1": 0.39, "tempo": 0.33},
    notes="threshold=0.08, peak_distance=2",
)
# pred_path = submissions/EXP-002_20260611_143022_a3f1b2c/predictions.json
```

Upload the `predictions.json` inside the versioned directory to the challenge
server. Record the leaderboard result in `experiment_log.md`.

## Prediction Format

Keys must be file stems **without** the `.wav` extension.
```json
{
  "test01": {"onsets": [0.21, 0.84, 1.47], "beats": [0.84, 1.68], "tempo": [60.0, 120.0]},
  "test02": {"onsets": [...], "beats": [...], "tempo": [...]}
}
```
Tempo list: one or two values, lower estimate first.
A missing key scores 0.0 for all metrics on that file.

## Evaluation Metrics (mir_eval)

| Task | Metric | Tolerance | Function |
|------|--------|-----------|----------|
| Onset detection | F-measure | ±50 ms | `mir_eval.onset.f_measure(..., window=0.05)` |
| Beat tracking | F-measure | ±70 ms | `mir_eval.beat.f_measure(ref, est)` |
| Tempo estimation | p-score | ±8% | `mir_eval.tempo.detection(ref_tempi, ref_weight, est_tempi, tol=0.08)` |

Mean score = (onset F1 + beat F1 + tempo p-score) / 3.
Tempo GT format: `[t_lo, t_hi, w]` where `w` is the annotator weight for
`t_lo`. Load this correctly from `.tempo.gt` files — do not default to 0.5.

## Current Best Validated Scores

Scores are from standalone 127-file evaluation scripts (not notebook runs).

| Metric | Validation (127 files) | Leaderboard |
|--------|----------------------|-------------|
| Onset F1 | **0.8051** (EXP-011) | *fill in* |
| Beat F1 | **0.7215** (EXP-012) | *fill in* |
| Tempo p-score | **0.7698** (EXP-008) | *fill in* |
| Mean | **0.7655** (EXP-012) | *fill in* |

Progression: EXP-007 Mean=0.7325 → EXP-008 (comb_fusion tempo) 0.7568 →
EXP-010 (multiband onset) 0.7593 → EXP-011 (whitening) 0.7629 →
EXP-012 (beat octave-select) 0.7655.
EXP-012 config: onset multiband (n_bands=2, merge_tol_ms=15, threshold=0.026),
superflux_gamma=200, superflux_mu=3, whiten=True (decay=0.995, floor=0.10),
dp_transition_width=0.10, dp_transition_lambda=1.0, tempo_search_min=60,
tempo_method=comb_fusion, tempo_comb_harmonics=2,
beat_octave_select=True (gate=78).

## Experiment Tracking

Before starting any experiment:
1. Check `experiments/experiment_backlog.md` — is this already proposed?
2. Check `experiments/experiment_log.md` — has this already been tried?

After completing an experiment:
- Move the entry from backlog to log
- Fill in: git commit before/after, validation scores, leaderboard result,
  failure analysis, keep/reject decision, next actions

Backlog is ranked by expected leaderboard impact. Do not skip ranks without
explicit justification.

## Git

- Commit before starting each experiment: `git commit -m "start EXP-003: custom peak picker"`
- Commit after validating: `git commit -m "EXP-003: onset F1 0.64 (+0.02 vs baseline)"`
- Never commit bare `predictions.json` to the repo root or `submissions/`
- Clear notebook outputs before committing (`Cell > All Output > Clear`)
- Push after each successful experiment

## What NOT to Do

- Do not import `librosa.beat.beat_track` or `librosa.beat.tempo` for detection
- Do not import `librosa.onset.onset_detect` or `librosa.util.peak_pick`
- Do not call `librosa.beat.tempo` / `librosa.feature.rhythm.tempo` as your tempo estimator
- Do not write to `submissions/predictions.json` directly
- Do not mutate `config.*` inside a sweep loop without restoring the original values
- Do not use `except:` bare clauses — always catch specific exception types
- Do not run the submission cell before the parameter cell
- Do not use the test set for validation — it has no annotations and the server
  rate-limits to one submission per 6 hours