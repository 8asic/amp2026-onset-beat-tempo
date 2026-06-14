"""
Manual detection-function extraction for onset, beat, and tempo tasks.

Challenge-safe:
- no ready-made onset features
- no ready-made peak picking
- no ready-made tempo estimation
- no ready-made beat tracking

This file only computes frame-level detection functions.
Peak picking and final detectors should be implemented separately.
"""

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
from scipy.ndimage import gaussian_filter1d, maximum_filter1d

from .config import config
from .preprocessing import Preprocessor


@dataclass
class DetectionFunctionResult:
    """Container for a frame-level detection function."""
    values: np.ndarray
    times: np.ndarray
    method: str
    sr: int
    hop_length: int


class FeatureExtractor:
    """Compute manual detection functions from allowed spectrogram representations."""

    def __init__(self, cfg=None):
        self.cfg = cfg or config
        self.sr = self.cfg.audio.sample_rate
        self.hop_length = self.cfg.audio.hop_length

        self.preprocessor = Preprocessor(
            sr=self.cfg.audio.sample_rate,
            n_fft=self.cfg.audio.onset_fft_size,
            hop_length=self.cfg.audio.onset_hop_length,
        )

    @staticmethod
    def positive_difference(S: np.ndarray) -> np.ndarray:
        """
        Compute positive frame-to-frame difference.

        S shape: (frequency_or_filter_bands, frames)
        Returns same shape.
        """
        diff = np.diff(S, axis=1, prepend=S[:, :1])
        return np.maximum(diff, 0.0)

    @staticmethod
    def normalize(d: np.ndarray, eps: float = 1e-10) -> np.ndarray:
        """
        Normalize a detection function to [0, 1].
        """
        d = np.asarray(d, dtype=np.float32)

        d = d - np.min(d)
        max_val = np.max(d)

        if max_val > eps:
            d = d / max_val

        return d

    def smooth(self, d: np.ndarray, sigma: Optional[float] = None) -> np.ndarray:
        """
        Smooth detection function with a Gaussian filter.
        """
        if sigma is None:
            sigma = self.cfg.onset.smoothing_sigma

        if sigma is not None and sigma > 0:
            return gaussian_filter1d(d, sigma=sigma)

        return d

    def log_spectral_flux(self, y: np.ndarray) -> DetectionFunctionResult:
        """
        Manual spectral flux on log-magnitude STFT.

        Pipeline:
        audio -> STFT magnitude -> log compression -> positive differences -> sum
        """
        spec = self.preprocessor.compute_stft(y)

        positive_diff = self.positive_difference(spec.log_magnitude)
        flux = np.sum(positive_diff, axis=0)

        flux = self.smooth(flux)
        flux = self.normalize(flux)

        return DetectionFunctionResult(
            values=flux,
            times=spec.times,
            method="log_spectral_flux",
            sr=spec.sr,
            hop_length=spec.hop_length,
        )

    def log_mel_flux(
        self,
        y: np.ndarray,
        n_mels: int = 80,
        log_scale: float = 100.0,
    ) -> DetectionFunctionResult:
        """
        Manual spectral flux on log-mel spectrogram.

        This is close to the lecture's LogFiltSpecFlux idea:
        filterbank -> log compression -> positive frame differences -> sum.
        """
        mel = self.preprocessor.mel_spectrogram(
            y,
            n_mels=n_mels,
            fmin=self.cfg.audio.onset_fmin,
            fmax=self.cfg.audio.onset_fmax,
            use_power=False,
            log_scale=log_scale,
        )

        positive_diff = self.positive_difference(mel.log_filtered)
        flux = np.sum(positive_diff, axis=0)

        flux = self.smooth(flux)
        flux = self.normalize(flux)

        return DetectionFunctionResult(
            values=flux,
            times=mel.times,
            method="log_mel_flux",
            sr=mel.sr,
            hop_length=mel.hop_length,
        )

    def log_semitone_flux(
        self,
        y: np.ndarray,
        log_scale: float = 100.0,
    ) -> DetectionFunctionResult:
        """
        Manual spectral flux on log-semitone spectrogram.

        This is useful for LogFiltSpecFlux-style experiments.
        """
        semi = self.preprocessor.semitone_spectrogram(
            y,
            fmin=27.5,
            fmax=self.cfg.audio.onset_fmax,
            log_scale=log_scale,
        )

        positive_diff = self.positive_difference(semi.log_filtered)
        flux = np.sum(positive_diff, axis=0)

        flux = self.smooth(flux)
        flux = self.normalize(flux)

        return DetectionFunctionResult(
            values=flux,
            times=semi.times,
            method="log_semitone_flux",
            sr=semi.sr,
            hop_length=semi.hop_length,
        )

    def high_frequency_content(self, y: np.ndarray) -> DetectionFunctionResult:
        """
        Manual high-frequency content detection function.

        HFC gives larger weight to high-frequency bins, where transient energy
        is often more visible.
        """
        spec = self.preprocessor.compute_stft(y)

        n_bins = spec.magnitude.shape[0]
        weights = np.arange(n_bins, dtype=np.float32).reshape(-1, 1)

        hfc = np.sum(weights * spec.power, axis=0)

        hfc = self.smooth(hfc)
        hfc = self.normalize(hfc)

        return DetectionFunctionResult(
            values=hfc,
            times=spec.times,
            method="high_frequency_content",
            sr=spec.sr,
            hop_length=spec.hop_length,
        )

    def superflux_like(
        self,
        y: np.ndarray,
        n_mels: int = 80,
        max_size: int = 3,
        log_scale: float = 100.0,
    ) -> DetectionFunctionResult:
        """
        Manual SuperFlux-like detection function.

        This is not librosa SuperFlux. It implements the main idea manually:
        apply a maximum filter over frequency bands before computing positive
        spectral differences. This can reduce false peaks caused by vibrato.
        """
        mel = self.preprocessor.mel_spectrogram(
            y,
            n_mels=n_mels,
            fmin=self.cfg.audio.onset_fmin,
            fmax=self.cfg.audio.onset_fmax,
            use_power=False,
            log_scale=log_scale,
        )

        # Maximum filtering over frequency axis only.
        # This is generic signal processing, not peak picking.
        max_filtered = maximum_filter1d(
            mel.log_filtered,
            size=max_size,
            axis=0,
            mode="nearest",
        )

        positive_diff = self.positive_difference(max_filtered)
        flux = np.sum(positive_diff, axis=0)

        flux = self.smooth(flux)
        flux = self.normalize(flux)

        return DetectionFunctionResult(
            values=flux,
            times=mel.times,
            method="superflux_like",
            sr=mel.sr,
            hop_length=mel.hop_length,
        )

    def combined_flux(
        self,
        y: np.ndarray,
        weights: tuple[float, float] = (0.7, 0.3),
    ) -> DetectionFunctionResult:
        """
        Combine log-mel flux and HFC.

        Useful experiment:
        - log-mel flux is general-purpose
        - HFC can help with percussive transients
        """
        mel_flux = self.log_mel_flux(y)
        hfc = self.high_frequency_content(y)

        n = min(len(mel_flux.values), len(hfc.values))
        combined = (
            weights[0] * mel_flux.values[:n]
            + weights[1] * hfc.values[:n]
        )

        combined = self.smooth(combined)
        combined = self.normalize(combined)

        return DetectionFunctionResult(
            values=combined,
            times=mel_flux.times[:n],
            method="combined_flux",
            sr=mel_flux.sr,
            hop_length=mel_flux.hop_length,
        )

    def onset_strength(
        self,
        y: np.ndarray,
        method: Optional[
            Literal[
                "log_spectral_flux",
                "log_mel_flux",
                "log_semitone_flux",
                "hfc",
                "superflux_like",
                "combined_flux",
                "bandwise_log_mel_flux",
                "beat_flux",
            ]
        ] = None,
    ) -> DetectionFunctionResult:
        """
        Main interface for manually computed onset detection functions.
        """
        if method is None:
            method = self.cfg.onset.method

        if method in {"flux", "log_spectral_flux"}:
            return self.log_spectral_flux(y)

        if method in {"log_mel_flux", "mel_flux"}:
            return self.log_mel_flux(y)

        if method in {"log_semitone_flux", "semitone_flux"}:
            return self.log_semitone_flux(y)

        if method in {"hfc", "high_frequency_content"}:
            return self.high_frequency_content(y)

        if method in {"superflux", "superflux_like"}:
            return self.superflux_like(y)

        if method == "combined_flux":
            return self.combined_flux(y)
        
        if method in {"bandwise_log_mel_flux", "beat_flux"}:
            return self.bandwise_log_mel_flux(y)

        raise ValueError(f"Unknown onset-strength method: {method}")

    def onset_envelope(self, y: np.ndarray, hop_length: int = None) -> np.ndarray:
        """
        Compatibility wrapper for later beat/tempo code.

        Important:
        This does NOT call librosa.onset.onset_strength.
        It returns our manually computed detection function.
        """
        if hop_length is not None and hop_length != self.cfg.audio.onset_hop_length:
            raise ValueError(
                "Changing hop_length here is not supported yet. "
                "Use the configured onset_hop_length for consistent timing."
            )

        return self.onset_strength(y).values
    
    def bandwise_log_mel_flux(
        self,
        y: np.ndarray,
        n_mels: int = 80,
        log_scale: float = 100.0,
    ) -> DetectionFunctionResult:
        """
        Beat-oriented detection function using weighted low/mid/high
        log-mel spectral flux.

        Weights are read from config.beat.beat_flux_weights.
        """
        weights = getattr(
            self.cfg.beat,
            "beat_flux_weights",
            (0.45, 0.40, 0.15),
        )

        mel = self.preprocessor.mel_spectrogram(
            y,
            n_mels=n_mels,
            fmin=self.cfg.audio.onset_fmin,
            fmax=self.cfg.audio.onset_fmax,
            use_power=False,
            log_scale=log_scale,
        )

        diff = self.positive_difference(mel.log_filtered)

        low = np.sum(diff[:20, :], axis=0)
        mid = np.sum(diff[20:55, :], axis=0)
        high = np.sum(diff[55:, :], axis=0)

        beat_flux = (
            weights[0] * low
            + weights[1] * mid
            + weights[2] * high
        )

        beat_flux = self.smooth(beat_flux)
        beat_flux = self.normalize(beat_flux)

        return DetectionFunctionResult(
            values=beat_flux,
            times=mel.times,
            method="bandwise_log_mel_flux",
            sr=mel.sr,
            hop_length=mel.hop_length,
        )