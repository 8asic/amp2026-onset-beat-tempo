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
        """Onset detection: superflux + custom LFSF peak picking (L04 slide 64).

        Multiband mode (config.onset.multiband) runs the picker independently on
        each mel-band ODF and merges peaks across bands — recovers soft onsets
        that are salient in one band but buried in the all-band sum.
        """
        fps = sr / self.cfg.audio.onset_hop_length
        delta = self.cfg.onset.threshold

        if self.cfg.onset.multiband:
            bands = self.fe.superflux_bands(y, self.cfg.onset.n_bands)
            frames: list[int] = []
            for k in range(bands.shape[0]):
                frames.extend(self._pick(bands[k], fps, delta))
            peaks = self._merge_frames(frames, fps)
        else:
            strength = self.fe.onset_strength(y, method=self.cfg.onset.method)
            peaks = self._pick(strength, fps, delta)

        return librosa.frames_to_time(
            np.array(peaks, dtype=int), sr=sr,
            hop_length=self.cfg.audio.onset_hop_length
        )

    def _pick(self, strength: np.ndarray, fps: float, delta: float) -> list:
        """LFSF adaptive peak picker: local max + adaptive mean + min IOI."""
        N = len(strength)
        w_max = max(1, int(round(0.030 * fps)))
        w_avg = max(1, int(round(0.100 * fps)))
        wait  = max(1, int(round(0.050 * fps)))

        peaks: list[int] = []
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
        return peaks

    def _merge_frames(self, frames: list, fps: float) -> list:
        """Merge cross-band peaks within merge_tol_ms into single onsets."""
        if not frames:
            return []
        tol = max(1, int(round(self.cfg.onset.merge_tol_ms / 1000.0 * fps)))
        ordered = sorted(set(frames))
        merged = [ordered[0]]
        for f in ordered[1:]:
            if f - merged[-1] > tol:
                merged.append(f)
        return merged


class BeatTracker:
    """Track beats in audio."""
    
    def __init__(self, cfg=None):
        self.cfg = cfg or config
    
    def track(self, y: np.ndarray, sr: int, tempo: Optional[float] = None) -> Tuple[np.ndarray, float]:
        """Beat tracking: tight-window Gaussian DP trellis.

        Optional second pass (config.beat.beat_two_pass): re-run the DP with the
        lag implied by the realized median inter-beat interval. This self-corrects
        files where the supplied tempo was a few % off the period the beats
        actually settled on.
        """
        onset_env = self._beat_odf(y, sr, self.cfg.audio.beat_hop_length)
        fps = sr / self.cfg.audio.beat_hop_length
        N = len(onset_env)

        if tempo is None:
            tempo = self._estimate_tempo_from_env(onset_env, fps)

        lag_min = max(1, int(np.ceil(60.0 / self.cfg.beat.tempo_max * fps)))
        lag_max = int(np.floor(60.0 / self.cfg.beat.tempo_min * fps))
        lag = int(np.clip(int(round(60.0 * fps / tempo)), lag_min, lag_max))

        beats = self._dp_beats(onset_env, lag, N)

        if self.cfg.beat.beat_two_pass and len(beats) >= 4:
            ibi = float(np.median(np.diff(beats)))
            new_lag = int(np.clip(int(round(ibi)), lag_min, lag_max))
            # Only re-run if the realized period differs beyond the search window
            # but is still the same metrical level (guard against octave flips).
            rel = abs(new_lag - lag) / float(lag)
            if new_lag != lag and rel < 0.5:
                beats2 = self._dp_beats(onset_env, new_lag, N)
                if len(beats2) >= 4:
                    beats, lag = beats2, new_lag

        out_tempo = 60.0 * fps / lag
        return (
            librosa.frames_to_time(
                np.array(beats, dtype=int), sr=sr,
                hop_length=self.cfg.audio.beat_hop_length
            ),
            float(out_tempo),
        )

    def _dp_beats(self, onset_env: np.ndarray, lag: int, N: int) -> list:
        """Tight-window Gaussian DP trellis for a fixed period `lag` (frames)."""
        width = self.cfg.beat.dp_transition_width
        lam   = self.cfg.beat.dp_transition_lambda
        # Tight search window: only ±width around expected period.
        # Gaussian penalty: -lam * ((delta - lag) / (width * lag))^2
        lo_off = max(1, int(round(lag * (1.0 - width))))
        hi_off = int(round(lag * (1.0 + width)))
        sigma  = width * lag + 1e-6   # denominator for Gaussian

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
        return beats

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
        Estimate tempo from a log-mel spectral-flux salience curve.

        Search range [tempo_search_min, tempo_max] avoids strong measure-level
        AC peaks that caused the old estimator to predict ~34 BPM for 100–200
        BPM music. The salience method (config.beat.tempo_method) selects the
        primary period:
          - "argmax":      plain normalized autocorrelation peak (EXP-007)
          - "comb":        harmonic comb sum of AC at integer multiples of lag
          - "comb_fusion": comb AC × Fourier tempogram (EXP-008, default goal)
        Comb/fusion resolve metrical-level (×1.5, ×2, ×3) confusions that a bare
        AC argmax cannot. The stored _primary (always in search range) feeds the
        beat tracker.
        """
        hop = self.cfg.audio.beat_hop_length
        onset_env = self._tempo_odf(y, sr, hop)
        fps = sr / float(hop)
        N = len(onset_env)

        lags, sal = self._salience(onset_env, fps, N)
        best_lag = int(lags[int(np.argmax(sal))])
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

    def _tempo_odf(self, y: np.ndarray, sr: int, hop: int) -> np.ndarray:
        """Log-mel spectral flux ODF (same family as the beat activation)."""
        mel = librosa.feature.melspectrogram(
            y=y, sr=sr, hop_length=hop, n_mels=80,
            fmin=self.cfg.audio.onset_fmin, fmax=self.cfg.audio.onset_fmax,
        )
        log_mel = np.log1p(mel)
        diff = np.diff(log_mel, axis=1, prepend=log_mel[:, :1])
        onset_env = np.sum(np.maximum(diff, 0), axis=0)
        if onset_env.max() > 0:
            onset_env /= onset_env.max()
        return onset_env

    def _salience(self, onset_env: np.ndarray, fps: float, N: int):
        """Return (lags, salience) over the tempo search range."""
        bpm_lo = self.cfg.beat.tempo_search_min   # 60.0
        bpm_hi = self.cfg.beat.tempo_max           # 200.0
        lag_min = max(1, int(np.ceil(60.0 / bpm_hi * fps)))
        lag_max = min(int(np.floor(60.0 / bpm_lo * fps)), N // 2)
        lags = np.arange(lag_min, lag_max + 1)

        ac = self._ac_full(onset_env, N)
        method = self.cfg.beat.tempo_method
        if method == "argmax":
            sal = ac[lags - 1].astype(np.float64)
        elif method == "comb":
            sal = self._comb_score(ac, lags)
        else:  # comb_fusion
            comb = self._comb_score(ac, lags)
            dft = self._dft_tempogram(onset_env, lags)
            sal = comb * dft
        return lags, sal

    @staticmethod
    def _ac_full(onset_env: np.ndarray, N: int) -> np.ndarray:
        """Normalized autocorrelation for all lags 1..N//2 (ac[k-1] = lag k)."""
        r = np.correlate(onset_env, onset_env, mode='full')[N - 1:]
        lags_all = np.arange(1, N // 2 + 1)
        ac = r[lags_all] / (np.float64(N) - lags_all.astype(np.float64) + 1e-10)
        ac = np.maximum(ac, 0.0)
        m = ac.max()
        if m > 0:
            ac = ac / m
        return ac

    def _comb_score(self, ac: np.ndarray, lags: np.ndarray) -> np.ndarray:
        """Sum AC at integer multiples h*lag — the true period accumulates the
        most harmonic support, so ×1.5/×2/×3 impostors lose."""
        H = self.cfg.beat.tempo_comb_harmonics
        L = len(ac)
        scores = np.zeros(len(lags), dtype=np.float64)
        for i, lag in enumerate(lags):
            s = 0.0
            for h in range(1, H + 1):
                idx = h * int(lag) - 1
                if idx < L:
                    s += ac[idx]
            scores[i] = s
        m = scores.max()
        if m > 0:
            scores /= m
        return scores

    @staticmethod
    def _dft_tempogram(onset_env: np.ndarray, lags: np.ndarray) -> np.ndarray:
        """Direct DFT magnitude at frequency 1/lag (cycles/frame) per candidate.
        AC over-favors long lags, the DFT over-favors short ones; their product
        suppresses both octave biases."""
        n = np.arange(len(onset_env), dtype=np.float64)
        env = onset_env - onset_env.mean()
        mags = np.empty(len(lags), dtype=np.float64)
        for i, lag in enumerate(lags):
            ang = 2.0 * np.pi * n / float(lag)
            re = float(np.dot(env, np.cos(ang)))
            im = float(np.dot(env, np.sin(ang)))
            mags[i] = np.hypot(re, im)
        m = mags.max()
        if m > 0:
            mags /= m
        return mags


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
        if self.cfg.beat.beat_octave_select:
            beats = self._track_best_octave(y, sr, beat_tempo)
        else:
            beats, _ = self.beat_tracker.track(y, sr, tempo=beat_tempo)

        return onsets, beats, tempos

    def _track_best_octave(self, y: np.ndarray, sr: int, base_bpm: float) -> np.ndarray:
        """Track beats at {base/2, base, base*2} and keep the grid whose beats sit
        on onset peaks far above the off-beat midpoints. A half-tempo grid skips
        real beats, so its midpoints land on onsets and contrast collapses; a
        double-tempo grid puts beats on quiet off-beats. The contrast is octave-
        fair, unlike raw onset sum/mean."""
        hop = self.cfg.audio.beat_hop_length
        odf = self.beat_tracker._beat_odf(y, sr, hop)
        fps = sr / hop
        lo, hi = self.cfg.beat.tempo_search_min, self.cfg.beat.tempo_max

        # Only very slow primaries are prone to the half-beat pathology; leave
        # plausible-beat tempos untouched (selecting an octave there regresses).
        if base_bpm >= self.cfg.beat.beat_octave_gate:
            beats, _ = self.beat_tracker.track(y, sr, tempo=base_bpm)
            return beats

        cands = []
        for mult in (1.0, 2.0):
            bpm = base_bpm * mult
            if lo <= bpm <= hi:
                cands.append(bpm)
        if not cands:
            cands = [float(np.clip(base_bpm, lo, hi))]

        best_beats, best_score = None, -np.inf
        for bpm in cands:
            beats, _ = self.beat_tracker.track(y, sr, tempo=bpm)
            frames = np.round(np.asarray(beats) * fps).astype(int)
            frames = frames[(frames >= 0) & (frames < len(odf))]
            if len(frames) < 2:
                continue
            on = float(odf[frames].mean())
            mids = (frames[:-1] + frames[1:]) // 2
            off = float(odf[mids].mean()) if len(mids) else 0.0
            score = on - off
            if score > best_score:
                best_beats, best_score = beats, score
        if best_beats is None:
            best_beats, _ = self.beat_tracker.track(y, sr, tempo=base_bpm)
        return best_beats