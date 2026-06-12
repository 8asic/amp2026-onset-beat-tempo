"""Feature extraction for onset, beat, and tempo detection."""

import numpy as np
import librosa
from scipy.ndimage import gaussian_filter1d, maximum_filter1d

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
        """
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