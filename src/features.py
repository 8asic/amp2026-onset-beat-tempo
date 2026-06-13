"""Feature extraction for onset, beat, and tempo detection."""

import numpy as np
import librosa
from scipy.ndimage import gaussian_filter1d, maximum_filter1d, median_filter

from .config import config


class FeatureExtractor:
    """Extract audio features for detection tasks."""
    
    def __init__(self, cfg=None):
        self.cfg = cfg or config
        self.sr = self.cfg.audio.sample_rate
        self.hop_length = self.cfg.audio.hop_length
    
    def spectral_flux(self, y: np.ndarray) -> np.ndarray:
        """
        Compute spectral flux onset strength.
        
        Returns:
            onset_strength array per frame
        """
        # Compute magnitude spectrogram
        S = np.abs(librosa.stft(
            y,
            n_fft=self.cfg.audio.onset_fft_size,
            hop_length=self.cfg.audio.onset_hop_length
        ))
        
        # Spectral flux: sum of positive differences
        diff = np.diff(S, axis=1, prepend=S[:, 0:1])
        flux = np.sum(diff * (diff > 0), axis=0)
        
        # Normalize
        if flux.max() > 0:
            flux = flux / flux.max()
        
        return flux
    
    def _superflux_posdiff(self, y: np.ndarray) -> np.ndarray:
        """Positive SuperFlux differences per mel bin (n_mels, n_frames).

        Shared core for `superflux()` and `superflux_bands()`. Optional adaptive
        spectral whitening (config.audio.whiten) divides each mel bin by a causal
        running peak before log compression, lifting soft onsets in quiet bands.
        When config.audio.hpss is True, percussive soft-masking is applied in the
        STFT domain (1025 bins) before projecting to mel for better separation of
        narrow harmonics vs broadband transients.
        """
        if self.cfg.audio.hpss:
            # STFT-domain HPSS: higher freq resolution than mel domain
            D = np.abs(librosa.stft(
                y, n_fft=self.cfg.audio.onset_fft_size,
                hop_length=self.cfg.audio.onset_hop_length,
            )) ** 2  # power spectrogram (n_fft//2+1, n_frames)
            kh = self.cfg.audio.hpss_kernel_harm
            kp = self.cfg.audio.hpss_kernel_perc
            harm = median_filter(D, size=(1, kh), mode='reflect')
            perc = median_filter(D, size=(kp, 1), mode='reflect')
            mask = perc / (perc + harm + 1e-8)
            mel_fb = librosa.filters.mel(
                sr=self.sr, n_fft=self.cfg.audio.onset_fft_size,
                n_mels=self.cfg.audio.onset_n_mels,
                fmin=self.cfg.audio.onset_fmin, fmax=self.cfg.audio.onset_fmax,
            )
            mel = np.dot(mel_fb, D * mask)
        else:
            mel = librosa.feature.melspectrogram(
                y=y,
                sr=self.sr,
                n_fft=self.cfg.audio.onset_fft_size,
                hop_length=self.cfg.audio.onset_hop_length,
                n_mels=self.cfg.audio.onset_n_mels,
                fmin=self.cfg.audio.onset_fmin,
                fmax=self.cfg.audio.onset_fmax,
            )
        if self.cfg.audio.whiten:
            mel = self._whiten(mel)
        log_mel = np.log1p(self.cfg.audio.superflux_gamma * mel)  # (n_mels, n_frames)
        # Max filter along frequency axis: vibrato shifts energy between adjacent
        # frequency bins at the same time instant, so comparing against the
        # per-bin neighbourhood max suppresses vibrato false positives.
        max_filt = maximum_filter1d(log_mel, size=self.cfg.audio.superflux_mu, axis=0)
        diff = log_mel[:, 1:] - max_filt[:, :-1]
        diff = np.pad(diff, ((0, 0), (1, 0)), mode='constant')
        return np.maximum(diff, 0.0)

    def _whiten(self, mel: np.ndarray) -> np.ndarray:
        """Adaptive whitening: divide each mel bin by a causal decaying running
        peak (floored at a fraction of the bin's global max so silent bins are
        not amplified into noise)."""
        decay = self.cfg.audio.whiten_decay
        floor_col = self.cfg.audio.whiten_floor * mel.max(axis=1)  # (n_mels,)
        psp = np.empty_like(mel)
        running = np.maximum(mel[:, 0], floor_col)
        psp[:, 0] = running
        for t in range(1, mel.shape[1]):
            running = np.maximum(mel[:, t], np.maximum(decay * running, floor_col))
            psp[:, t] = running
        return mel / psp

    def superflux(self, y: np.ndarray) -> np.ndarray:
        """Real SuperFlux onset strength (Böck et al. ICASSP 2012)."""
        odf = np.sum(self._superflux_posdiff(y), axis=0)
        if odf.max() > 0:
            odf /= odf.max()
        return odf
    
    def superflux_bands(self, y: np.ndarray, n_bands: int) -> np.ndarray:
        """Per-band Real SuperFlux ODFs.

        Same SuperFlux computation as `superflux()`, but instead of summing the
        positive flux across all mel bins it sums within `n_bands` contiguous
        mel-band groups, returning one normalized ODF per band. A soft onset
        salient in only one band survives here even when louder energy in other
        bands would swamp it in the all-band sum.
        """
        pos = self._superflux_posdiff(y)  # (n_mels, n_frames)
        n_mels = pos.shape[0]
        edges = np.linspace(0, n_mels, n_bands + 1).astype(int)
        bands = np.empty((n_bands, pos.shape[1]), dtype=np.float64)
        for b in range(n_bands):
            seg = pos[edges[b]:edges[b + 1], :].sum(axis=0)
            m = seg.max()
            bands[b] = seg / m if m > 0 else seg

        if self.cfg.onset.smoothing_sigma > 0:
            bands = gaussian_filter1d(bands, sigma=self.cfg.onset.smoothing_sigma, axis=1)
        return bands

    def _onset_stft(self, y: np.ndarray) -> np.ndarray:
        """Complex STFT at the onset hop (shared by complex/HFC/phase ODFs)."""
        return librosa.stft(
            y,
            n_fft=self.cfg.audio.onset_fft_size,
            hop_length=self.cfg.audio.onset_hop_length,
        )

    @staticmethod
    def _princarg(x: np.ndarray) -> np.ndarray:
        """Wrap phase to (-pi, pi]."""
        return np.mod(x + np.pi, 2.0 * np.pi) - np.pi

    @staticmethod
    def _norm(odf: np.ndarray) -> np.ndarray:
        m = odf.max()
        return odf / m if m > 0 else odf

    def complex_domain(self, y: np.ndarray, X: np.ndarray = None) -> np.ndarray:
        """Rectified complex-domain ODF (Bello/Duxbury).

        Predict each bin from the previous frame assuming constant magnitude and
        constant phase rate, then measure the complex deviation of the actual
        frame. Rectified: only bins whose magnitude rises contribute, so it fires
        on energy increases (onsets) not decays. Catches soft/tonal onsets that
        magnitude-only flux misses.
        """
        if X is None:
            X = self._onset_stft(y)
        mag = np.abs(X)
        phase = np.angle(X)
        # Predicted phase for frame t: linear extrapolation 2*phi[t-1]-phi[t-2]
        phase_pred = 2.0 * phase[:, 1:-1] - phase[:, :-2]
        X_pred = mag[:, 1:-1] * np.exp(1j * phase_pred)   # predicts frames 2..T-1
        dev = np.abs(X[:, 2:] - X_pred)
        rising = mag[:, 2:] >= mag[:, 1:-1]               # rectify to energy rises
        cd = np.sum(dev * rising, axis=0)
        cd = np.pad(cd, (2, 0), mode='constant')          # align to length T
        return self._norm(cd)

    def hfc(self, y: np.ndarray, X: np.ndarray = None) -> np.ndarray:
        """High-frequency-content flux ODF (Masri).

        HFC[t] = sum_k k * |X[k,t]|^2 emphasises high-frequency energy, which
        spikes on percussive/broadband attacks. ODF is the positive first
        difference of HFC.
        """
        if X is None:
            X = self._onset_stft(y)
        mag2 = np.abs(X) ** 2
        k = np.arange(mag2.shape[0], dtype=np.float64)[:, None]
        hfc = np.sum(k * mag2, axis=0)
        d = np.diff(hfc, prepend=hfc[:1])
        return self._norm(np.maximum(d, 0.0))

    def phase_deviation(self, y: np.ndarray, X: np.ndarray = None) -> np.ndarray:
        """Magnitude-weighted phase-deviation ODF (Bello).

        At a steady tone the phase advances linearly, so the second phase
        difference is ~0; at an onset it jumps. Weighting by magnitude suppresses
        noisy phase in silent bins. Complements energy-based ODFs on tonal onsets.
        """
        if X is None:
            X = self._onset_stft(y)
        mag = np.abs(X)
        phase = np.angle(X)
        ddphi = self._princarg(phase[:, 2:] - 2.0 * phase[:, 1:-1] + phase[:, :-2])
        num = np.sum(mag[:, 2:] * np.abs(ddphi), axis=0)
        den = np.sum(mag[:, 2:], axis=0) + 1e-9
        pd = num / den
        pd = np.pad(pd, (2, 0), mode='constant')
        return self._norm(pd)

    def onset_channels(self, y: np.ndarray) -> np.ndarray:
        """Stack of normalized onset ODFs for fusion peak-picking.

        config.onset.fusion_odfs lists which channels to include. "superflux"
        expands to per-band SuperFlux ODFs (config.onset.n_bands); the others are
        single full-spectrum curves. Each channel is smoothed and normalized so
        the per-channel peak picker sees comparable scales; OnsetDetector picks
        peaks per channel and merges across channels (same as multiband).
        """
        names = self.cfg.onset.fusion_odfs
        X = None
        if any(n in ("complex", "hfc", "phase") for n in names):
            X = self._onset_stft(y)
        sigma = self.cfg.onset.smoothing_sigma

        def smooth(odf: np.ndarray) -> np.ndarray:
            return gaussian_filter1d(odf, sigma=sigma) if sigma > 0 else odf

        chans: list[np.ndarray] = []
        for name in names:
            if name == "superflux":
                # superflux_bands already smooths + normalizes each band
                chans.extend(self.superflux_bands(y, self.cfg.onset.n_bands))
            elif name == "complex":
                chans.append(smooth(self.complex_domain(y, X)))
            elif name == "hfc":
                chans.append(smooth(self.hfc(y, X)))
            elif name == "phase":
                chans.append(smooth(self.phase_deviation(y, X)))
            else:
                raise ValueError(f"unknown fusion ODF: {name}")
        return np.vstack(chans)

    def onset_strength(self, y: np.ndarray, method: str = "superflux") -> np.ndarray:
        """Compute onset strength using specified method."""
        if method == "flux":
            strength = self.spectral_flux(y)
        else:
            strength = self.superflux(y)
        
        # Apply smoothing
        if self.cfg.onset.smoothing_sigma > 0:
            strength = gaussian_filter1d(strength, sigma=self.cfg.onset.smoothing_sigma)
        
        return strength
    
    def onset_envelope(self, y: np.ndarray, hop_length: int = None) -> np.ndarray:
        """Compute onset envelope for beat tracking."""
        if hop_length is None:
            hop_length = self.cfg.audio.beat_hop_length
        
        return librosa.onset.onset_strength(
            y=y,
            sr=self.sr,
            hop_length=hop_length
        )