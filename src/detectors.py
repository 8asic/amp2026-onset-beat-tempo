"""Onset, beat, and tempo detection algorithms."""

import numpy as np
import librosa
from pathlib import Path
from typing import Tuple, Optional, List

from .config import config
from .features import FeatureExtractor
from .learned_onset import LearnedOnsetModel


class OnsetDetector:
    """Detect note onsets in audio."""
    
    def __init__(self, cfg=None):
        self.cfg = cfg or config
        self.fe = FeatureExtractor(cfg)
        self._learned_model = None  # lazily loaded LearnedOnsetModel (EXP-016)
        self._onset_cnn = None      # lazily loaded EXP-020 PyTorch onset CNN
        self._cnn_act_cache: Optional[np.ndarray] = None  # EXP-029: last raw activation for beat blending

    def detect(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Onset detection: onset-strength path feeds the SAME hand-written picker.

        Paths (first available wins):
        - cnn (EXP-020): PyTorch CNN activation over log-mel (needs torch+weights)
        - learned (EXP-016): pure-numpy LR activation over ODF channels
        - fusion (EXP-015): per-ODF-channel picking + cross-channel merge
        - multiband (EXP-010): per-mel-band superflux picking + merge
        The picker (`_pick`) always makes the musical decision.
        """
        fps = sr / self.cfg.audio.onset_hop_length
        delta = self.cfg.onset.threshold

        if self.cfg.onset.cnn:
            act = self._onset_cnn_activation(y, sr)
            if act is not None:
                peaks = self._pick(act, fps, self.cfg.onset.cnn_delta)
                return librosa.frames_to_time(
                    np.array(peaks, dtype=int), sr=sr,
                    hop_length=self.cfg.audio.onset_hop_length,
                )
            # torch/weights unavailable -> fall through to fusion below

        if self.cfg.onset.learned:
            model = self._get_learned_model()
            chans = self.fe.onset_channels(y)  # uses config.onset.fusion_odfs
            act = model.predict_activation(chans)
            peaks = self._pick(act, fps, self.cfg.onset.learned_delta)
        elif self.cfg.onset.fusion:
            chans = self.fe.onset_channels(y)
            frames: list[int] = []
            for k in range(chans.shape[0]):
                frames.extend(self._pick(chans[k], fps, delta))
            peaks = self._merge_frames(frames, fps)
        elif self.cfg.onset.multiband:
            bands = self.fe.superflux_bands(y, self.cfg.onset.n_bands)
            frames = []
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

    def _get_learned_model(self) -> LearnedOnsetModel:
        """Lazily load the EXP-016 learned onset model and verify channel match."""
        if self._learned_model is None:
            path = self.cfg.onset.learned_model_path
            if not Path(path).is_absolute():
                path = self.cfg.paths.base_dir / path
            self._learned_model = LearnedOnsetModel.load(path)
            if tuple(self.cfg.onset.fusion_odfs) != self._learned_model.odfs:
                raise ValueError(
                    f"learned model ODF channels {self._learned_model.odfs} "
                    f"!= config.onset.fusion_odfs {tuple(self.cfg.onset.fusion_odfs)}; "
                    "set them equal so inference features match training."
                )
        return self._learned_model

    def _load_onset_cnn(self):
        """Lazily load the EXP-020 onset CNN checkpoint (torch optional).

        Returns a dict {net, torch, mu, sd, hop, n_fft, n_mels, fmin, fmax} or
        None if torch is unavailable or the weights file is missing (then the
        caller falls back to the fusion onset). Rebuilds the small 16/32/32 CNN.
        """
        if self._onset_cnn is not None:
            return self._onset_cnn
        path = self.cfg.onset.cnn_model_path
        if not Path(path).is_absolute():
            path = self.cfg.paths.base_dir / path
        if not Path(path).exists():
            return None
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            return None

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        sd = ckpt["state_dict"]
        # Infer conv channels, INPUT channels (1=single-res, 3=multi-res EXP-023),
        # and head width from the checkpoint -> loads any trained variant.
        in_ch = sd["conv.0.weight"].shape[1]
        c1 = sd["conv.0.weight"].shape[0]
        c2 = sd["conv.4.weight"].shape[0]
        c3 = sd["conv.8.weight"].shape[0]
        hidden = sd["head.0.weight"].shape[0]

        class OnsetCNN(nn.Module):
            def __init__(self, n_mels):
                super().__init__()
                self.conv = nn.Sequential(
                    nn.Conv2d(in_ch, c1, (3, 3), padding=(1, 1)), nn.BatchNorm2d(c1), nn.ReLU(),
                    nn.MaxPool2d((1, 3)),
                    nn.Conv2d(c1, c2, (3, 3), padding=(1, 1)), nn.BatchNorm2d(c2), nn.ReLU(),
                    nn.MaxPool2d((1, 3)),
                    nn.Conv2d(c2, c3, (3, 3), padding=(1, 1)), nn.BatchNorm2d(c3), nn.ReLU(),
                )
                with torch.no_grad():
                    f = self.conv(torch.zeros(1, in_ch, 8, n_mels)).shape[-1]
                self.head = nn.Sequential(
                    nn.Linear(c3 * f, hidden), nn.ReLU(), nn.Dropout(0.3), nn.Linear(hidden, 1))

            def forward(self, x):                 # x: (B, in_ch, T, n_mels)
                h = self.conv(x)
                B, C, T, F = h.shape
                h = h.permute(0, 2, 1, 3).reshape(B, T, C * F)
                return self.head(h).squeeze(-1)

        net = OnsetCNN(int(ckpt["n_mels"]))
        net.load_state_dict(sd)
        net.eval()
        n_ffts = ckpt.get("n_ffts")
        if n_ffts is None:
            n_ffts = [int(ckpt["n_fft"])]
        self._onset_cnn = {
            "net": net, "torch": torch, "in_ch": int(in_ch),
            "mu": np.asarray(ckpt["mu"], dtype=np.float64),
            "sd": np.asarray(ckpt["sd"], dtype=np.float64),
            "n_mels": int(ckpt["n_mels"]), "hop": int(ckpt["hop"]),
            "n_ffts": [int(n) for n in n_ffts],
            "fmin": float(ckpt["fmin"]), "fmax": float(ckpt["fmax"]),
        }
        return self._onset_cnn

    def _onset_cnn_activation(self, y: np.ndarray, sr: int):
        """Per-frame onset activation from the CNN; None if model unavailable.

        Handles single-res (1 n_fft) and multi-res (EXP-023: 3 STFT windows
        stacked as input channels). EXP-028: optional pitch-shift test-time
        augmentation (config.audio.onset_tta_shifts) averages the activation over
        pitch-shifted copies (pitch preserves timing, so frames align) — synergises
        with the pitch-augmented CNN to reduce variance.
        """
        m = self._load_onset_cnn()
        if m is None:
            return None
        shifts = self.cfg.audio.onset_tta_shifts or (0,)
        acts = []
        for sh in shifts:
            ys = librosa.effects.pitch_shift(y, sr=sr, n_steps=sh) if sh != 0 else y
            acts.append(self._cnn_forward(ys, sr, m))
        T = min(len(a) for a in acts)
        result = np.mean([a[:T] for a in acts], axis=0)
        self._cnn_act_cache = result  # EXP-029: expose for beat blending in Pipeline
        return result

    def _cnn_forward(self, y: np.ndarray, sr: int, m) -> np.ndarray:
        """Single forward pass of the onset CNN on audio y -> activation."""
        specs = []
        for nfft in m["n_ffts"]:
            mel = librosa.feature.melspectrogram(
                y=y, sr=sr, n_fft=nfft, hop_length=m["hop"], n_mels=m["n_mels"],
                fmin=m["fmin"], fmax=m["fmax"],
            )
            specs.append(np.log1p(mel).T)             # (T, n_mels)
        T = min(s.shape[0] for s in specs)
        feat = np.stack([s[:T] for s in specs], axis=0)   # (in_ch, T, n_mels)
        feat = (feat - m["mu"]) / m["sd"]
        torch = m["torch"]
        with torch.no_grad():
            t = torch.from_numpy(feat.astype(np.float32)).unsqueeze(0)
            return torch.sigmoid(m["net"](t)).squeeze(0).numpy().astype(np.float64)


class BeatTracker:
    """Track beats in audio."""

    def __init__(self, cfg=None):
        self.cfg = cfg or config
        self._beat_model = None             # EXP-018: lazily loaded BLSTM checkpoint
        self._odf_cache = (None, None)      # (id(y), activation) to avoid recompute

    def track(self, y: np.ndarray, sr: int, tempo: Optional[float] = None,
              onset_act: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
        """Beat tracking: tight-window Gaussian DP trellis.

        Optional second pass (config.beat.beat_two_pass): re-run the DP with the
        lag implied by the realized median inter-beat interval. This self-corrects
        files where the supplied tempo was a few % off the period the beats
        actually settled on.

        onset_act: optional downsampled onset CNN activation (EXP-029) to blend
        into the beat ODF, sharpening phase precision.
        """
        onset_env = self._beat_odf(y, sr, self.cfg.audio.beat_hop_length)
        # EXP-029C: mix onset CNN activation into beat ODF for sharper phase snap
        if onset_act is not None and self.cfg.beat.onset_blend > 0.0:
            alpha = self.cfg.beat.onset_blend
            T = min(len(onset_env), len(onset_act))
            onset_env = onset_env.copy()
            onset_env[:T] = (1.0 - alpha) * onset_env[:T] + alpha * onset_act[:T]
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

    def _dbn_beats(self, onset_env: np.ndarray, fps: float, tempo_prior: float) -> list:
        """Joint tempo+phase beat decoder (simplified DBN / variable-lag Viterbi).

        State = "a beat at frame t whose incoming inter-beat interval is lag".
        Unlike the fixed-lag DP, the lag may DRIFT frame-to-frame (penalised by
        dbn_lam_change), so it tracks tempo changes/rubato; it is also anchored to
        the comb_fusion tempo_prior (dbn_lam_prior) to keep the strong global
        tempo. Vectorised O(N·L²) Viterbi over (frame, lag). Our own code — no
        librosa.beat, no madmom.
        """
        N = len(onset_env)
        a = onset_env.astype(np.float64)
        lag_min = max(2, int(np.ceil(60.0 / self.cfg.beat.tempo_max * fps)))
        lag_max = int(np.floor(60.0 / self.cfg.beat.tempo_min * fps))
        lag_max = min(lag_max, N - 1)
        if lag_max <= lag_min:
            return list(range(0, N, max(1, lag_min)))
        lags = np.arange(lag_min, lag_max + 1)
        L = len(lags)

        lam_ch = self.cfg.beat.dbn_lam_change
        sig_ch = self.cfg.beat.dbn_sigma_change * float(lags.mean()) + 1e-9
        Lk = lags.reshape(-1, 1).astype(np.float64)
        Lj = lags.reshape(1, -1).astype(np.float64)
        Tc = -lam_ch * ((Lj - Lk) / sig_ch) ** 2          # (L,L): Tc[k,j] from lag k -> j
        TcT = Tc.T.copy()                                 # (L,L): [j,k]

        pr = np.zeros(L)
        if tempo_prior and tempo_prior > 0:
            lag_prior = 60.0 * fps / tempo_prior
            sig_pr = self.cfg.beat.dbn_sigma_prior * lag_prior + 1e-9
            pr = -self.cfg.beat.dbn_lam_prior * ((lags - lag_prior) / sig_pr) ** 2

        NEG = -1e18
        score = np.full((N, L), NEG)
        back = np.full((N, L), -1, dtype=int)
        for t in range(0, min(lag_min, N)):               # earliest beats: no predecessor
            score[t] = a[t] + pr

        for t in range(lag_min, N):
            idx = t - lags                                # (L,) predecessor frame per target lag j
            prev = np.full((L, L), NEG)
            ok = idx >= 0
            if ok.any():
                prev[ok] = score[idx[ok]]                 # prev[j] = score[t-lags[j]]
            cand = prev + TcT                             # cand[j,k]
            k = cand.argmax(axis=1)
            mx = cand[np.arange(L), k]
            score[t] = a[t] + pr + mx
            back[t] = k
            if t < lag_max:                               # also allow starting here
                start = a[t] + pr
                better = start > score[t]
                score[t][better] = start[better]
                back[t][better] = -1

        lo = max(0, N - lag_max)                          # last beat near the end
        best, bt, bj = NEG, N - 1, 0
        for t in range(lo, N):
            j = int(np.argmax(score[t]))
            if score[t, j] > best:
                best, bt, bj = score[t, j], t, j

        beats: list[int] = []
        t, j = bt, bj
        while t >= 0 and j >= 0:
            beats.append(t)
            k = back[t, j]
            if k < 0:
                break
            t = t - lags[j]
            j = k
        beats.sort()
        return beats

    def _beat_odf(self, y: np.ndarray, sr: int, hop_length: int) -> np.ndarray:
        """Beat-tracking activation feeding the DP/tempo.

        EXP-018: if config.beat.learned and the BLSTM weights load, return the
        learned per-frame beat activation; otherwise fall back to log-mel spectral
        flux. The downstream tempo+DP (and octave-select) are identical either way,
        so the learned model only upgrades the beat *signal*.
        """
        if self.cfg.beat.learned:
            act = self._blstm_activation(y, sr, hop_length)
            if act is not None:
                return act
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

    def _load_beat_model(self):
        """Lazily load the EXP-018 BLSTM checkpoint (torch optional).

        Returns a dict {model, mu, sd, hop, n_fft, n_mels, fmin, fmax} or None if
        torch is unavailable or the weights file is missing (then we fall back to
        log-mel flux). Reconstructs the same BiLSTM architecture used in training.
        """
        if self._beat_model is not None:
            return self._beat_model
        path = self.cfg.beat.learned_model_path
        if not Path(path).is_absolute():
            path = self.cfg.paths.base_dir / path
        if not Path(path).exists():
            return None
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            return None

        ckpt = torch.load(path, map_location="cpu", weights_only=False)

        class BeatBLSTM(nn.Module):
            def __init__(self, n_mels, hidden, layers):
                super().__init__()
                self.lstm = nn.LSTM(n_mels, hidden, layers, batch_first=True,
                                    bidirectional=True,
                                    dropout=0.3 if layers > 1 else 0.0)
                self.fc = nn.Linear(2 * hidden, 1)

            def forward(self, x):
                h, _ = self.lstm(x)
                return self.fc(h).squeeze(-1)

        net = BeatBLSTM(ckpt["n_mels"], ckpt["hidden"], ckpt["layers"])
        net.load_state_dict(ckpt["state_dict"])
        net.eval()
        self._beat_model = {
            "net": net, "torch": torch,
            "mu": np.asarray(ckpt["mu"], dtype=np.float64),
            "sd": np.asarray(ckpt["sd"], dtype=np.float64),
            "hop": int(ckpt["hop"]), "n_fft": int(ckpt["n_fft"]),
            "n_mels": int(ckpt["n_mels"]),
            "fmin": float(ckpt["fmin"]), "fmax": float(ckpt["fmax"]),
        }
        return self._beat_model

    def _blstm_activation(self, y: np.ndarray, sr: int, hop_length: int):
        """Per-frame beat activation from the BLSTM; None if model unavailable."""
        key = id(y)
        if self._odf_cache[0] == key:
            return self._odf_cache[1]
        m = self._load_beat_model()
        if m is None:
            return None
        mel = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=m["n_fft"], hop_length=m["hop"], n_mels=m["n_mels"],
            fmin=m["fmin"], fmax=m["fmax"],
        )
        log_mel = np.log1p(mel).T                       # (T, n_mels)
        feat = (log_mel - m["mu"]) / m["sd"]
        torch = m["torch"]
        with torch.no_grad():
            t = torch.from_numpy(feat.astype(np.float32)).unsqueeze(0)
            act = torch.sigmoid(m["net"](t)).squeeze(0).numpy().astype(np.float64)
        self._odf_cache = (key, act)
        return act

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

        # EXP-025 tempo ensemble: if comb_fusion and the beat-activation AC tempo
        # DISAGREE (>8%), offer both primaries as the submission pair (covers
        # octave/meter errors); if they agree, keep the comb_fusion octave pair.
        if self.cfg.beat.tempo_ensemble:
            hop = self.cfg.audio.beat_hop_length
            env = self.beat_tracker._beat_odf(y, sr, hop)
            t_beat = self.beat_tracker._estimate_tempo_from_env(env, sr / hop)
            if abs(beat_tempo - t_beat) / max(beat_tempo, 1e-9) >= 0.08:
                tempos = sorted([float(beat_tempo), float(t_beat)])

        # Detect onsets
        onsets = self.onset_detector.detect(y, sr)

        # EXP-029C: extract downsampled onset CNN activation for beat ODF blending.
        # onset hop=256, beat hop=512 → downsample by 2 via pair-averaging.
        _onset_act_ds: Optional[np.ndarray] = None
        if self.cfg.beat.onset_blend > 0.0:
            raw = self.onset_detector._cnn_act_cache
            if raw is not None and len(raw) >= 2:
                n = (len(raw) // 2) * 2
                _onset_act_ds = (raw[:n:2] + raw[1:n:2]) / 2.0

        # Track beats. EXP-018 decode "A": derive tempo from the beat activation
        # itself (autocorrelation) and skip octave-select — the activation already
        # encodes period. Decode "B" (default) keeps comb_fusion tempo + octave-
        # select and just swaps the beat ODF for the learned activation.
        if self.cfg.beat.decoder == "dbn":
            # EXP-024: joint tempo+phase DBN decode over the beat activation,
            # anchored to the comb_fusion tempo prior.
            hop = self.cfg.audio.beat_hop_length
            env = self.beat_tracker._beat_odf(y, sr, hop)
            fps = sr / hop
            frames = self.beat_tracker._dbn_beats(env, fps, beat_tempo)
            beats = librosa.frames_to_time(np.array(frames, dtype=int), sr=sr,
                                           hop_length=hop)
        elif self.cfg.beat.learned and self.cfg.beat.learned_decode == "A":
            beats, _ = self.beat_tracker.track(y, sr, tempo=None, onset_act=_onset_act_ds)
        elif self.cfg.beat.beat_octave_select:
            beats = self._track_best_octave(y, sr, beat_tempo, onset_act=_onset_act_ds)
        else:
            beats, _ = self.beat_tracker.track(y, sr, tempo=beat_tempo, onset_act=_onset_act_ds)

        # EXP-029B: IBI-derived tempo as a third oracle.
        # After we have the final beat grid, compute median IBI → BPM. If it
        # disagrees with ALL current estimates (>8%), replace/extend the pair.
        if self.cfg.beat.ibi_tempo and len(beats) >= 6:
            ibis = np.diff(beats)
            med = float(np.median(ibis))
            # Outlier filter: keep IBIs within [0.4, 2.5] × median
            clean = ibis[(ibis > 0.4 * med) & (ibis < 2.5 * med)]
            if len(clean) >= 4:
                ibi_bpm = 60.0 / float(np.median(clean))
                # Octave-normalise into plausible tempo range
                while ibi_bpm < 60.0:
                    ibi_bpm *= 2.0
                while ibi_bpm > 200.0:
                    ibi_bpm /= 2.0
                agrees = any(abs(ibi_bpm - t) / max(t, 1e-9) < 0.08 for t in tempos)
                if not agrees:
                    # IBI oracle sees something comb_fusion missed — offer both
                    tempos = sorted([float(tempos[0]), float(ibi_bpm)])

        return onsets, beats, tempos

    def _track_best_octave(self, y: np.ndarray, sr: int, base_bpm: float,
                           onset_act: Optional[np.ndarray] = None) -> np.ndarray:
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
            beats, _ = self.beat_tracker.track(y, sr, tempo=base_bpm, onset_act=onset_act)
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
            beats, _ = self.beat_tracker.track(y, sr, tempo=bpm, onset_act=onset_act)
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
            best_beats, _ = self.beat_tracker.track(y, sr, tempo=base_bpm, onset_act=onset_act)
        return best_beats