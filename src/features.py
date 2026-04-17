"""Feature extraction for onset, beat, and tempo detection."""

import numpy as np
import librosa
from scipy.ndimage import gaussian_filter1d

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
        """
        Compute superflux onset strength (better for complex music).
        
        Uses mel-scaled spectrogram with positive differences per band.
        """
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=self.sr,
            n_fft=self.cfg.audio.onset_fft_size,
            hop_length=self.cfg.audio.onset_hop_length,
            n_mels=64,
            fmin=self.cfg.audio.onset_fmin,
            fmax=self.cfg.audio.onset_fmax
        )
        
        # Log compression
        mel_spec = np.log1p(mel_spec)
        
        # Positive differences per band, then sum
        diff = np.diff(mel_spec, axis=1, prepend=mel_spec[:, 0:1])
        positive_diff = np.maximum(diff, 0)
        superflux_val = np.sum(positive_diff, axis=0)
        
        # Normalize
        if superflux_val.max() > 0:
            superflux_val = superflux_val / superflux_val.max()
        
        return superflux_val
    
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