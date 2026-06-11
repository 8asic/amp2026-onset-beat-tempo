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
        """Beat tracking: tight-window Gaussian DP trellis."""
        onset_env = self._beat_odf(y, sr, self.cfg.audio.beat_hop_length)
        fps = sr / self.cfg.audio.beat_hop_length
        N = len(onset_env)

        if tempo is None:
            tempo = self._estimate_tempo_from_env(onset_env, fps)

        lag_min = max(1, int(np.ceil(60.0 / self.cfg.beat.tempo_max * fps)))
        lag_max = int(np.floor(60.0 / self.cfg.beat.tempo_min * fps))
        lag = int(np.clip(int(round(60.0 * fps / tempo)), lag_min, lag_max))

        width = self.cfg.beat.dp_transition_width
        lam   = self.cfg.beat.dp_transition_lambda
        # Tight search window: only ±width around expected period.
        # Gaussian penalty: -lam * ((delta - lag) / (width * lag))^2
        lo_off = max(1, int(round(lag * (1.0 - width))))
        hi_off = int(round(lag * (1.0 + width)))
        sigma  = width * lag + 1e-6   # denominator for Gaussian

        # Precompute penalty for the steady-state window (hi_off - lo_off + 1 elements).
        # Deltas run from hi_off down to lo_off as we scan from lo to hi.
        steady_deltas = np.arange(hi_off, lo_off - 1, -1, dtype=float)
        trans_buf = -lam * ((steady_deltas - lag) / sigma) ** 2
        window_size = hi_off - lo_off + 1

        score = onset_env.copy().astype(float)
        back  = np.arange(N, dtype=int)

        for t in range(1, N):
            lo = max(0, t - hi_off)
            hi = t - lo_off
            if hi < 0 or lo > hi:
                back[t] = max(0, t - lag)
                continue
            n     = hi - lo + 1
            cands = score[lo : hi + 1]
            if n == window_size:
                combined = cands + trans_buf
            else:
                deltas   = t - np.arange(lo, hi + 1, dtype=float)
                combined = cands + (-lam * ((deltas - lag) / sigma) ** 2)
            best       = int(np.argmax(combined))
            back[t]    = lo + best
            score[t]   = onset_env[t] + combined[best]

        t = int(np.argmax(score))
        beats: list[int] = []
        visited: set[int] = set()
        while t not in visited:
            visited.add(t)
            beats.append(t)
            t = int(back[t])

        beats.sort()
        return (
            librosa.frames_to_time(
                np.array(beats, dtype=int), sr=sr,
                hop_length=self.cfg.audio.beat_hop_length
            ),
            float(tempo),
        )

    def _beat_odf(self, y: np.ndarray, sr: int, hop_length: int) -> np.ndarray:
        """Log-mel spectral flux: positive first differences of log-mel spectrogram."""
        mel = librosa.feature.melspectrogram(
            y=y, sr=sr, hop_length=hop_length, n_mels=80,
            fmin=self.cfg.audio.onset_fmin, fmax=self.cfg.audio.onset_fmax,
        )
        log_mel = np.log1p(mel)
        diff = np.diff(log_mel, axis=1, prepend=log_mel[:, :1])
        flux = np.sum(np.maximum(diff, 0), axis=0)
        if flux.max() > 0:
            flux /= flux.max()
        return flux

    def _estimate_tempo_from_env(self, onset_env: np.ndarray, fps: float) -> float:
        """Autocorrelation tempo estimation (L05 slide 23)."""
        N = len(onset_env)
        lag_min = max(1, int(np.ceil(60.0 / self.cfg.beat.tempo_max * fps)))
        lag_max = min(int(np.floor(60.0 / self.cfg.beat.tempo_min * fps)), N // 2)
        lags = np.arange(lag_min, lag_max + 1)
        r = np.correlate(onset_env, onset_env, mode='full')
        r = r[N - 1:]
        denom = np.float64(N) - lags.astype(np.float64) + 1e-10
        r_norm = r[lags] / denom
        best_lag = int(lags[np.argmax(r_norm)])
        return 60.0 * fps / best_lag


class TempoEstimator:
    """Estimate tempo from audio."""

    def __init__(self, cfg=None):
        self.cfg = cfg or config
        self._primary: float = 120.0  # last AC primary (for beat tracker)

    def estimate(self, y: np.ndarray, sr: int) -> List[float]:
        """
        Estimate tempo using log-mel spectral flux autocorrelation.

        Search range [tempo_search_min, tempo_max] avoids strong measure-level
        AC peaks that cause the old librosa-based estimator to predict ~34 BPM
        for 100–200 BPM music.  The stored _primary (always in search range) is
        used by the multi-hypothesis beat tracker.
        """
        hop = self.cfg.audio.beat_hop_length
        mel = librosa.feature.melspectrogram(
            y=y, sr=sr, hop_length=hop, n_mels=80,
            fmin=self.cfg.audio.onset_fmin, fmax=self.cfg.audio.onset_fmax,
        )
        log_mel = np.log1p(mel)
        diff = np.diff(log_mel, axis=1, prepend=log_mel[:, :1])
        onset_env = np.sum(np.maximum(diff, 0), axis=0)
        if onset_env.max() > 0:
            onset_env /= onset_env.max()

        fps = sr / float(hop)
        N = len(onset_env)

        bpm_lo = self.cfg.beat.tempo_search_min   # 60.0
        bpm_hi = self.cfg.beat.tempo_max           # 200.0
        lag_min = max(1, int(np.ceil(60.0 / bpm_hi * fps)))
        lag_max = min(int(np.floor(60.0 / bpm_lo * fps)), N // 2)

        lags = np.arange(lag_min, lag_max + 1)
        r = np.correlate(onset_env, onset_env, mode='full')[N - 1:]
        denom = np.float64(N) - lags.astype(np.float64) + 1e-10
        r_norm = r[lags] / denom

        best_lag = int(lags[int(np.argmax(r_norm))])
        primary = 60.0 * fps / best_lag
        self._primary = float(primary)

        # Submission pair: primary + its octave alternative, sorted [lo, hi].
        bpm_min = self.cfg.beat.tempo_min
        bpm_max = self.cfg.beat.tempo_max
        if primary * 2.0 <= bpm_max:
            return [float(primary), float(primary * 2.0)]
        elif primary / 2.0 >= bpm_min:
            return [float(primary / 2.0), float(primary)]
        return [float(primary)]


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
        # Estimate tempo first; _primary is always in [tempo_search_min, tempo_max]
        tempos = self.tempo_estimator.estimate(y, sr)
        # Use the raw AC primary for beat tracking — avoids sending half-tempo to the DP
        # when primary > 100 BPM (tempos[0] would be primary/2 in that case).
        beat_tempo = self.tempo_estimator._primary

        # Detect onsets
        onsets = self.onset_detector.detect(y, sr)

        # Track beats using estimated tempo
        beats, _ = self.beat_tracker.track(y, sr, tempo=beat_tempo)
        
        return onsets, beats, tempos