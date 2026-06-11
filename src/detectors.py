"""Onset, beat, and tempo detection algorithms."""

import numpy as np
import librosa
from scipy.signal import find_peaks
from typing import Tuple, Optional, List

from .config import config
from .features import FeatureExtractor


class OnsetDetector:
    """Detect note onsets in audio."""
    
    def __init__(self, cfg=None):
        self.cfg = cfg or config
        self.fe = FeatureExtractor(cfg)
    
    def detect(self, y: np.ndarray, sr: int) -> np.ndarray:
        """
        Detect onsets using superflux detection function + custom LFSF peak picking.
        Implements the three conditions from L04 slide 64 (Böck et al. 2012).
        """
        strength = self.fe.onset_strength(y, method=self.cfg.onset.method)
        fps = sr / self.cfg.audio.onset_hop_length
        N = len(strength)

        # Window sizes in frames, scaled to hop size (1 frame ≈ 23ms at 512/22050)
        # pre_max/post_max: local maximum window (~30ms each side)
        # pre_avg/post_avg: mean window for adaptive threshold (~100ms each side)
        # wait:             minimum IOI (~50ms = eval window tolerance)
        w_max = max(1, int(round(0.030 * fps)))   # ~30ms
        w_avg = max(1, int(round(0.100 * fps)))   # ~100ms
        wait  = max(1, int(round(0.050 * fps)))   # ~50ms

        delta = self.cfg.onset.threshold

        peaks = []
        last_onset = -wait - 1

        for n in range(N):
            x = strength[n]

            # Condition 1: local maximum in window [n-w_max, n+w_max]
            lo = max(0, n - w_max)
            hi = min(N, n + w_max + 1)
            if x < strength[lo:hi].max():
                continue

            # Condition 2: greater than mean in window [n-w_avg, n+w_avg] plus delta
            lo_avg = max(0, n - w_avg)
            hi_avg = min(N, n + w_avg + 1)
            if x < strength[lo_avg:hi_avg].mean() + delta:
                continue

            # Condition 3: minimum distance from last accepted onset
            if n - last_onset <= wait:
                continue

            peaks.append(n)
            last_onset = n

        return librosa.frames_to_time(
            np.array(peaks, dtype=int),
            sr=sr,
            hop_length=self.cfg.audio.onset_hop_length
        )


class BeatTracker:
    """Track beats in audio."""
    
    def __init__(self, cfg=None):
        self.cfg = cfg or config
    
    def track(self, y: np.ndarray, sr: int, tempo: Optional[float] = None) -> Tuple[np.ndarray, float]:
        """Beat tracking via autocorrelation (L05 slides 20-31)."""
        onset_env = librosa.onset.onset_strength(
            y=y, sr=sr, hop_length=self.cfg.audio.beat_hop_length
        )
        fps = sr / self.cfg.audio.beat_hop_length
        N = len(onset_env)

        if tempo is None:
            tempo = self._estimate_tempo_from_env(onset_env, fps)

        best_lag = max(1, int(round(60.0 * fps / tempo)))
        lag_min = max(1, int(np.ceil(60.0 / self.cfg.beat.tempo_max * fps)))
        lag_max = int(np.floor(60.0 / self.cfg.beat.tempo_min * fps))
        best_lag = int(np.clip(best_lag, lag_min, lag_max))

        scores = np.array([
            onset_env[np.arange(ph, N, best_lag)].sum()
            for ph in range(best_lag)
        ])
        best_phase = int(np.argmax(scores))

        beat_frames = np.arange(best_phase, N, best_lag)
        beat_times = librosa.frames_to_time(
            beat_frames, sr=sr, hop_length=self.cfg.audio.beat_hop_length
        )
        return beat_times, float(tempo)

    def _estimate_tempo_from_env(self, onset_env: np.ndarray, fps: float) -> float:
        """Autocorrelation tempo estimation (L05 slide 23)."""
        N = len(onset_env)
        lag_min = max(1, int(np.ceil(60.0 / self.cfg.beat.tempo_max * fps)))
        lag_max = min(int(np.floor(60.0 / self.cfg.beat.tempo_min * fps)), N // 2)
        lags = np.arange(lag_min, lag_max + 1)
        r = np.correlate(onset_env, onset_env, mode='full')
        r = r[N - 1:]
        r_norm = r[lags] / (N - lags.astype(float) + 1e-10)
        best_lag = int(lags[np.argmax(r_norm)])
        return 60.0 * fps / best_lag


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
        """
        Process a single audio file.
        
        Returns:
            (onsets, beats, tempos)
        """
        # Estimate tempo first
        tempos = self.tempo_estimator.estimate(y, sr)
        primary_tempo = tempos[0]
        
        # Detect onsets
        onsets = self.onset_detector.detect(y, sr)
        
        # Track beats using estimated tempo
        beats, _ = self.beat_tracker.track(y, sr, tempo=primary_tempo)
        
        return onsets, beats, tempos