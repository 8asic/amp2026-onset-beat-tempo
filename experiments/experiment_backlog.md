# Experiment Backlog — AMP Challenge

> **Rules:** Proposed experiments only. Move to `experiment_log.md` once
> started. Check this file before proposing new experiments to avoid
> duplicating work. Ranked by expected leaderboard impact.

---

## Priority Order

| Rank | ID | Title | Expected Gain | Confidence | Effort | Risk |
|------|----|-------|---------------|------------|--------|------|
| 1 | EXP-016 | Pure-numpy LEARNED onset activation (logistic/MLP on ODF features incl. complex-domain) → existing peak picker; k-fold held-out eval on 277 files | +0.01–0.03 onset (277) | Medium | Moderate (4–6h) | Medium (train-on-test if eval sloppy) |
| 2 | EXP-017 | Residual tempo: beat-level multi-hypothesis for the 26 ambiguous-meter files | +0.01–0.03 tempo & beat | Medium | Moderate | Medium (overfit-prone) |
| 3 | EXP-018 | Pure-numpy beat activation model (BLSTM-lite/TCN, hand-written BPTT) → existing DP | +0.03–0.08 beat | Medium | Hard (15–20h) | Medium |

**Constraint:** neural models must be pure-numpy (PyTorch/TF not on the allowed
list; madmom RNNs explicitly banned). Peak-picking/DP decision stays our own code.

**Done (now in experiment_log.md):** EXP-004 (γ/μ sweep), EXP-005 (tight DP),
EXP-006 (log-mel beat ODF), EXP-007 (tempo search floor), EXP-008 (comb_fusion
tempo), EXP-009 (two-pass beat — REJECTED no-op), EXP-010 (multiband onset),
EXP-011 (adaptive whitening), EXP-012 (beat octave-select), EXP-013 (beat phase —
REJECTED no-op), EXP-014 (HPSS — REJECTED train overfit; established 277-file
onset screening rule), EXP-015 (multi-ODF fusion superflux+complex — KEEP,
onset 277 0.7493→0.7615).

---

## EXP-002 — Dynamic Programming Beat Tracker [DONE — see experiment_log.md]

### Objective
Replace the current uniform-grid autocorrelation beat tracker in
`BeatTracker.track()` with a Viterbi-style DP trellis that can adapt to
tempo changes and is pulled toward high onset energy frames.

### Rationale
Beat F1 is stuck at 0.3791 across all threshold values in EXP-001.
The current tracker computes a single tempo lag from autocorrelation, then
places beats at `phase + k * lag` for k=0,1,2,... This rigid grid fails when:
- Tempo changes mid-file
- There are long silences (no onset energy to track)
- The correct tempo requires sub-beat subdivision (half-bar periodicity)

The DP approach from Ellis (2007) / Böck & Schedl (2011) maintains a
score function over beat positions and uses a transition cost that penalises
deviations from the expected inter-beat interval.

### Algorithm Sketch (Ellis 2007)
```
C(t) = onset_env(t) + max_{t' < t} [ C(t') + W(t - t', lag) ]
W(δ, lag) = -α * (log(δ/lag))²  # penalty for tempo deviation
Backtrack C to get beat sequence.
```
`α` controls tightness (higher = more regular). Typical α = 400.

Source: D. Ellis "Beat tracking by dynamic programming", J. New Music Res. 2007.
Also: L05 slides 35–40 (DP tracker derivation).

### Expected Gain
+0.10–0.20 beat F1. The gain is large because the current uniform grid
can be completely wrong for pieces with tempo changes or pickups.

### Confidence
High — well-understood algorithm with established implementation recipes.

### Effort
Hard (~4–6 hours). The DP requires O(N²) transitions unless vectorised;
need to implement efficient max over prior beats with log-Gaussian penalty.

### Risk
Medium. The tightness parameter α needs tuning. Miscalibrated α can make
beats irregular (too low) or too rigid (too high). Must validate on 127 files
before submitting. Beat F1 could temporarily drop if the phase estimation is
wrong.

### Files to Modify
**`src/config.py`** — add `BeatConfig.dp_alpha: float = 400.0` (tightness).

**`src/detectors.py`** — replace `BeatTracker.track()`:

```python
def track(self, y, sr, tempo=None):
    onset_env = librosa.onset.onset_strength(
        y=y, sr=sr, hop_length=self.cfg.audio.beat_hop_length
    )
    fps = sr / self.cfg.audio.beat_hop_length
    N = len(onset_env)

    if tempo is None:
        tempo = self._estimate_tempo_from_env(onset_env, fps)

    lag = max(1, int(round(60.0 * fps / tempo)))
    alpha = self.cfg.beat.dp_alpha

    # DP trellis: score[t] = best cumulative beat score ending at frame t
    score = onset_env.copy()
    backtrack = np.zeros(N, dtype=int)

    for t in range(1, N):
        # Search window: beats from 0.5*lag to 2*lag before t
        lo = max(0, t - 2 * lag)
        hi = max(0, t - lag // 2)
        if lo >= hi:
            backtrack[t] = max(0, t - lag)
            continue
        candidates = np.arange(lo, hi)
        deltas = t - candidates
        transition = -alpha * (np.log(deltas / lag)) ** 2
        combined = score[candidates] + transition
        best = int(np.argmax(combined))
        backtrack[t] = candidates[best]
        score[t] = onset_env[t] + combined[best]

    # Backtrack from best final frame
    beats = []
    t = int(np.argmax(score))
    while t > 0:
        beats.append(t)
        prev = backtrack[t]
        if prev == t:
            break
        t = prev
    beats = sorted(beats)
    beat_times = librosa.frames_to_time(
        np.array(beats, dtype=int), sr=sr,
        hop_length=self.cfg.audio.beat_hop_length
    )
    return beat_times, float(tempo)
```

### Success Criteria
Beat F1 ≥ 0.45 on 127-file validation (vs current 0.3791, +0.07 minimum).

---

## EXP-004 — γ and μ Sweep on 127 Files

### Objective
Sweep `superflux_gamma` over [50, 100, 200] and `superflux_mu` over [1, 2, 3, 5]
on the full 127-file validation set to confirm the best settings.

### Rationale
The preprocessing notebook (`00_preprocessing.ipynb`) tested these parameters
on a 5-file sample. Key findings:
- γ=200 gave higher F1 than γ=100 at threshold=0.01 on 5 files (0.7622 vs 0.7595)
- μ=1 gave 0.7735 vs μ=3 gave 0.7654 — but μ=1 is just standard log-mel flux
  (the max filter of size 1 is identity). This may not generalise to 127 files.
- These 5-file results have high variance. Need 127-file confirmation.

### Expected Gain
+0.01–0.03 onset F1.

### Confidence
High for γ sweep. Lower for μ (the 5-file μ=1 result might be noise).

### Effort
Easy — add to Sweep cell in `01_pipeline.ipynb`. ~30 min runtime.

### Risk
None — read-only sweep, config restored after.

### Files to Modify
Add sweep cell to `notebooks/01_pipeline.ipynb`. Update `src/config.py`
default if a better value is found.

### Success Criteria
Identify γ and μ values that improve onset F1 ≥ 0.79 on 127 files.

---

## EXP-005 — 99th-Percentile ODF Normalisation

### Objective
In `FeatureExtractor.superflux()`, replace `odf /= odf.max()` with
`odf /= np.percentile(odf, 99)` followed by `np.clip(odf, 0, None)`.

### Rationale
`max` normalisation makes the ODF scale dominated by the single loudest event.
A 99th-percentile divisor clips the top 1% (preventing one loud drum hit from
compressing all other onsets) and spreads the remaining 99% more evenly.
Effect is strongest on files with one or two outlier-loud events.

### Expected Gain
+0.01–0.03 onset F1 on files with highly non-uniform onset amplitudes.

### Confidence
Medium — depends on the amplitude distribution of training files.

### Effort
Easy — 2-line change in `src/features.py`.

### Risk
Low. If onset F1 drops, revert.

### Files to Modify
**`src/features.py`** — `superflux()` normalisation block:
```python
# BEFORE:
if odf.max() > 0:
    odf /= odf.max()

# AFTER:
if odf.max() > 0:
    p99 = np.percentile(odf, 99)
    odf /= (p99 if p99 > 0 else odf.max())
    odf = np.clip(odf, 0, None)
```

### Success Criteria
Onset F1 ≥ 0.785 on 127-file validation (vs current 0.7824, any improvement counts).

---

## EXP-006 — HPSS Soft-Mask + Real SuperFlux at hop=256

### Objective
Add HPSS-based percussive isolation before the mel spectrogram step in
`superflux()`. Compute a Wiener soft mask (percussive component = median filter
along time axis, harmonic = along frequency axis), apply to the audio, then
compute Real SuperFlux on the masked signal.

### Rationale
The preprocessing notebook Exp D (Combined = pre-emphasis α=0.97 + HPSS +
Real SuperFlux) at hop=512 gave F1=0.7778. At hop=256, Real SuperFlux alone
gives F1=0.7801 — they're already close. The Combined approach at hop=256 was
not tested and could be meaningfully better.

HPSS separates percussive transients (broadband, short in time, hence
picked up by a time-median filter) from harmonic content (narrow-band, long
in time). For onset detection, working on the percussive component reduces
harmonic modulation false positives.

### Expected Gain
+0.01–0.02 onset F1 on top of current 0.7824.

### Confidence
Medium — the 5-file preprocessing result was small (+0.003 for Combined vs
Real SuperFlux at hop=512). May be noise.

### Effort
Moderate — requires adding HPSS to `superflux()` or a new `superflux_hpss()`
method plus config toggle.

### Risk
Low. HPSS adds compute time (~2× slower per file) but does not change
correctness.

### Files to Modify
**`src/config.py`** — add `AudioConfig.superflux_hpss: bool = False`.

**`src/features.py`** — add HPSS branch in `superflux()` when
`self.cfg.audio.superflux_hpss` is True.

### Success Criteria
Onset F1 ≥ 0.790 (vs 0.7824 baseline for this component).

---

## EXP-007 — Multi-Hypothesis Beat Tracker (Fixed Winner Selection)

### Objective
Retry multi-hypothesis beat tracking with a better winner scoring function.
Instead of `mean(onset_env[beats])`, use total DP score normalised by
expected beat count (`max(score) / (N / lag)`), which removes the half-tempo
bias identified in EXP-003.

### Rationale
EXP-003 showed multi-hypothesis can help (ff123_ATrain: 0.030→0.200,
ff123_BigYellow: 0.387→0.655) but the winner selection was flawed — it
preferred half-tempo hypotheses because sparse beats on strong onset frames
score higher per beat than dense beats on all frames.

The correct scorer should reward hypotheses that collect MORE total onset
energy (not higher per-beat energy). Options:
- `max(score) / (N/lag)`: total DP score per "expected beat opportunity"
- `max(score) * lag / N`: equivalent re-arrangement

### Expected Gain
+0.02–0.05 beat F1 (if winner selection works; EXP-003 lost 0.062).

### Confidence
Medium — the theory is sound but the exact scorer needs empirical validation.

### Effort
Moderate — modify `_dp_run` to return `(beats, max_score)` and update `track()`.

### Risk
Medium — need to re-validate on 127 files.

### Files to Modify
`src/detectors.py` — `BeatTracker.track()` and `_dp_run()`.

### Success Criteria
Beat F1 ≥ 0.51 on 127 files (strictly better than EXP-002's 0.4960).

---

## EXP-008 — Windowed Tempo Re-Estimation

### Objective
Instead of estimating a single global tempo for the whole file, re-estimate
tempo every W seconds using a sliding window, then use the local tempo to
update the beat grid periodically.

### Rationale
The current beat tracker computes one lag from the full-file autocorrelation.
For pieces with tempo change or drift, the global lag is a compromise that fits
neither the beginning nor end well. A windowed approach computes lag(t) at each
window position and adjusts the beat placement accordingly.

### Expected Gain
+0.02–0.06 beat F1 on files with tempo changes. May hurt stable-tempo files
if the window estimate is noisy.

### Confidence
Medium — depends on what fraction of test files have tempo changes.

### Effort
Moderate — ~3 hours.

### Risk
Medium. Noisy windowed estimates can introduce beat doubling/halving errors.
Needs careful blending between windows.

### Success Criteria
Beat F1 ≥ 0.41 on 127 files without introducing regressions on high-F1 files.

---

## EXP-009 — Use Real SuperFlux Envelope for Beat Tracking

### Objective
Replace `librosa.onset.onset_strength` in `BeatTracker.track()` and
`TempoEstimator.estimate()` with the custom `FeatureExtractor.onset_strength()`
using Real SuperFlux.

### Rationale
The beat tracker and tempo estimator currently use `librosa.onset.onset_strength`
(allowed as a feature extractor). Our Real SuperFlux is better calibrated for
this dataset (F1=0.7824 vs what librosa's default gives). Using a consistent
onset envelope throughout the pipeline might improve beat/tempo coherence.

### Expected Gain
+0.01–0.04 beat F1. Potentially also +0.01–0.02 tempo.

### Confidence
Low — onset strength for beat tracking is not the same task as for onset
detection. The right ODF for beat tracking may be different (e.g., more
harmonic-sensitive).

### Effort
Easy — swap `librosa.onset.onset_strength` calls in `detectors.py` with
`self.fe.onset_strength()`.

### Risk
Low. Easy to revert.

### Success Criteria
Beat F1 improves by ≥ 0.01 vs current (0.3791).
