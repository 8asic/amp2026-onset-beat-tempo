"""Onset, beat, and tempo detection algorithms."""

import numpy as np
import librosa
from typing import Tuple, Optional, List

from .config import config
from .features import FeatureExtractor


class OnsetDetector:
    """Detect note onsets in audio."""
    
    def __init__(self, cfg=None):
        self.cfg = cfg or config
        self.fe = FeatureExtractor(cfg)
    
    def detect(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Onset detection: superflux + custom LFSF peak picking (L04 slide 64)."""
        strength = self.fe.onset_strength(y, method=self.cfg.onset.method)
        fps = sr / self.cfg.audio.onset_hop_length
        N = len(strength)

        w_max = max(1, int(round(0.030 * fps)))
        w_avg = max(1, int(round(0.100 * fps)))
        wait  = max(1, int(round(0.050 * fps)))
        delta = self.cfg.onset.threshold

        peaks = []
        last_onset = -wait - 1

        for n in range(N):
            x = strength[n]
            lo = max(0, n - w_max)
            hi = min(N, n + w_max + 1)
            if x < strength[lo:hi].max():
                continue
            lo_avg = max(0, n - w_avg)
            hi_avg = min(N, n + w_avg + 1)
            if x < strength[lo_avg:hi_avg].mean() + delta:
                continue
            if n - last_onset <= wait:
                continue
            peaks.append(n)
            last_onset = n

        return librosa.frames_to_time(
            np.array(peaks, dtype=int), sr=sr,
            hop_length=self.cfg.audio.onset_hop_length
        )


class BeatTracker:
    """Track beats in audio."""
    
    def __init__(self, cfg=None):
        self.cfg = cfg or config
    
    def track(self, y: np.ndarray, sr: int, tempo: Optional[float] = None) -> Tuple[np.ndarray, float]:
        """Beat tracking: multi-hypothesis DP (Ellis 2007) over top-K AC candidates."""
        onset_env = librosa.onset.onset_strength(
            y=y, sr=sr, hop_length=self.cfg.audio.beat_hop_length
        )
        fps = sr / self.cfg.audio.beat_hop_length
        N = len(onset_env)

        lag_min = max(1, int(np.ceil(60.0 / self.cfg.beat.tempo_max * fps)))
        lag_max = int(np.floor(60.0 / self.cfg.beat.tempo_min * fps))

        # Candidate lags: top-K AC local maxima + external hint + octave alternatives.
        # Using multiple starting points lets the DP escape a wrong initial tempo.
        base_lags = self._ac_top_k_lags(onset_env, fps, top_k=4)
        if tempo is not None:
            base_lags.append(int(np.clip(round(60.0 * fps / tempo), lag_min, lag_max)))

        candidate_lags: set[int] = set()
        for lag in base_lags:
            lag = int(np.clip(lag, lag_min, lag_max))
            candidate_lags.add(lag)
            if lag // 2 >= lag_min:
                candidate_lags.add(lag // 2)
            if lag * 2 <= lag_max:
                candidate_lags.add(lag * 2)

        best_beats: list[int] = []
        best_score = -np.inf
        best_lag = min(candidate_lags)

        for lag in candidate_lags:
            beats = self._dp_run(onset_env, N, lag)
            if beats:
                s = float(np.mean(onset_env[np.array(beats)]))
                if s > best_score:
                    best_score = s
                    best_beats = beats
                    best_lag = lag

        if not best_beats:
            best_beats = [0]

        return (
            librosa.frames_to_time(
                np.array(best_beats, dtype=int), sr=sr,
                hop_length=self.cfg.audio.beat_hop_length
            ),
            60.0 * fps / best_lag,
        )

    def _dp_run(self, onset_env: np.ndarray, N: int, lag: int) -> list[int]:
        """Run DP trellis for one lag hypothesis; return beat frame indices."""
        alpha = self.cfg.beat.dp_alpha
        lo_off = max(1, lag // 2)
        window_size = 2 * lag - lo_off + 1

        trans_buf = -alpha * np.log(
            np.arange(2 * lag, lo_off - 1, -1, dtype=float) / lag
        ) ** 2

        score = onset_env.copy().astype(float)
        back = np.arange(N, dtype=int)

        for t in range(1, N):
            lo = max(0, t - 2 * lag)
            hi = t - lo_off
            if hi < 0 or lo > hi:
                back[t] = max(0, t - lag)
                continue
            n = hi - lo + 1
            cands = score[lo : hi + 1]
            combined = (cands + trans_buf if n == window_size
                        else cands + (-alpha * np.log(
                            (t - np.arange(lo, hi + 1, dtype=float)) / lag) ** 2))
            best = int(np.argmax(combined))
            back[t] = lo + best
            score[t] = onset_env[t] + combined[best]

        t = int(np.argmax(score))
        beats: list[int] = []
        visited: set[int] = set()
        while t not in visited:
            visited.add(t)
            beats.append(t)
            t = int(back[t])
        beats.sort()
        return beats

    def _ac_top_k_lags(self, onset_env: np.ndarray, fps: float, top_k: int = 4) -> list[int]:
        """Top-k beat lags from autocorrelation local maxima (vectorised, no library)."""
        N = len(onset_env)
        lag_min = max(1, int(np.ceil(60.0 / self.cfg.beat.tempo_max * fps)))
        lag_max = min(int(np.floor(60.0 / self.cfg.beat.tempo_min * fps)), N // 2)
        lags = np.arange(lag_min, lag_max + 1)

        r = np.correlate(onset_env, onset_env, mode='full')
        r_norm = r[N - 1:][lags] / (N - lags.astype(float) + 1e-10)

        # Local maxima via vectorised comparison — arithmetic, not a musical decision
        is_peak = np.zeros(len(r_norm), dtype=bool)
        is_peak[1:-1] = (r_norm[1:-1] > r_norm[:-2]) & (r_norm[1:-1] > r_norm[2:])
        peak_lags = lags[is_peak]
        peak_vals = r_norm[is_peak]

        if len(peak_lags) == 0:
            return [int(lags[np.argmax(r_norm)])]

        top_idx = np.argsort(peak_vals)[::-1][:top_k]
        return [int(peak_lags[i]) for i in top_idx]


class TempoEstimator:
    """Estimate tempo from audio."""
    
    def __init__(self, cfg=None):
        self.cfg = cfg or config
    
    def estimate(self, y: np.ndarray, sr: int) -> List[float]:
        """
        Estimate tempo using autocorrelation of onset envelope (L05 slides 23-24).
        """
        onset_env = librosa.onset.onset_strength(
            y=y, sr=sr, hop_length=self.cfg.audio.tempo_hop_length
        )
        fps = sr / self.cfg.audio.tempo_hop_length
        N = len(onset_env)

        bpm_min = self.cfg.beat.tempo_min
        bpm_max = self.cfg.beat.tempo_max
        lag_min = max(1, int(np.ceil(60.0 / bpm_max * fps)))
        lag_max = min(int(np.floor(60.0 / bpm_min * fps)), N // 2)

        lags = np.arange(lag_min, lag_max + 1)

        r = np.correlate(onset_env, onset_env, mode='full')
        r = r[N - 1:]
        r_norm = r[lags] / (N - lags.astype(float) + 1e-10)

        best_idx = int(np.argmax(r_norm))
        best_lag = int(lags[best_idx])
        primary_bpm = 60.0 * fps / best_lag

        # Return primary tempo + octave alternative
        if primary_bpm * 2 <= self.cfg.beat.tempo_max:
            return [float(primary_bpm), float(primary_bpm * 2)]
        elif primary_bpm / 2 >= self.cfg.beat.tempo_min:
            return [float(primary_bpm / 2), float(primary_bpm)]
        
        return [float(primary_bpm)]


class Pipeline:
    """Complete pipeline combining all detectors."""
    
    def __init__(self, cfg=None):
        self.cfg = cfg or config
        self.onset_detector = OnsetDetector(cfg)
        self.beat_tracker = BeatTracker(cfg)
        self.tempo_estimator = TempoEstimator(cfg)
    
    def process_file(self, y: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray, List[float]]:
        """Process one file: onset detection + multi-hypothesis beat/tempo."""
        onsets = self.onset_detector.detect(y, sr)

        # Beat tracker finds the best lag across multiple AC hypotheses;
        # tempo is derived from that lag (coherent with the returned beats).
        beats, beat_bpm = self.beat_tracker.track(y, sr)

        if beat_bpm * 2 <= self.cfg.beat.tempo_max:
            tempos = [float(beat_bpm), float(beat_bpm * 2)]
        elif beat_bpm / 2 >= self.cfg.beat.tempo_min:
            tempos = [float(beat_bpm / 2), float(beat_bpm)]
        else:
            tempos = [float(beat_bpm)]

        return onsets, beats, tempos