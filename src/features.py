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
    
    def superflux(self, y: np.ndarray) -> np.ndarray:
        """Real SuperFlux onset strength (Böck et al. ICASSP 2012)."""
        mel = librosa.feature.melspectrogram(
            y=y,
            sr=self.sr,
            n_fft=self.cfg.audio.onset_fft_size,
            hop_length=self.cfg.audio.onset_hop_length,
            n_mels=self.cfg.audio.onset_n_mels,
            fmin=self.cfg.audio.onset_fmin,
            fmax=self.cfg.audio.onset_fmax,
        )
        log_mel = np.log1p(self.cfg.audio.superflux_gamma * mel)  # (n_mels, n_frames)
        # Max filter along frequency axis: vibrato shifts energy between adjacent
        # frequency bins at the same time instant, so comparing against the
        # per-bin neighbourhood max suppresses vibrato false positives.
        max_filt = maximum_filter1d(log_mel, size=self.cfg.audio.superflux_mu, axis=0)
        diff = log_mel[:, 1:] - max_filt[:, :-1]
        diff = np.pad(diff, ((0, 0), (1, 0)), mode='constant')
        odf = np.sum(np.maximum(diff, 0), axis=0)
        if odf.max() > 0:
            odf /= odf.max()
        return odf
    
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