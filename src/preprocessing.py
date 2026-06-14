"""
Preprocessing utilities for the AMP 2026 challenge.

Allowed by challenge rules:
- audio reading / resampling
- STFT / spectrograms
- mel or semitone spectrograms
- generic signal processing and linear algebra

Not included here:
- onset detection
- peak picking
- tempo estimation
- beat tracking
"""

from dataclasses import dataclass
from typing import Literal, Optional

import librosa
import numpy as np


@dataclass
class SpectrogramResult:
    """Container for frame-based audio representation."""
    magnitude: np.ndarray          # shape: (freq_bins, frames)
    power: np.ndarray              # shape: (freq_bins, frames)
    log_magnitude: np.ndarray      # shape: (freq_bins, frames)
    times: np.ndarray              # shape: (frames,)
    sr: int
    n_fft: int
    hop_length: int


@dataclass
class FilterbankResult:
    """Container for mel/semitone-like filtered spectrogram."""
    filtered: np.ndarray           # shape: (bands, frames)
    log_filtered: np.ndarray       # shape: (bands, frames)
    times: np.ndarray              # shape: (frames,)
    sr: int
    n_fft: int
    hop_length: int
    scale: str


class Preprocessor:
    """
    Responsible only for allowed preprocessing:
    loading, normalization, STFT, log compression, and filterbank projection.
    """

    def __init__(
        self,
        sr: int = 44100,
        n_fft: int = 1024,
        hop_length: int = 441,
        win_length: Optional[int] = None,
        window: str = "hann",
        center: bool = True,
        eps: float = 1e-10,
    ):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length or n_fft
        self.window = window
        self.center = center
        self.eps = eps

    def load_audio(self, path: str) -> tuple[np.ndarray, int]:
        """
        Load audio as mono and resample to target sr.

        librosa.load is allowed because the challenge permits audio reading
        and resampling.
        """
        y, sr = librosa.load(path, sr=self.sr, mono=True)
        y = self.normalize_audio(y)
        return y, sr

    def normalize_audio(self, y: np.ndarray) -> np.ndarray:
        """
        Peak-normalize audio to [-1, 1] range without changing silence.

        This is not a task-specific audio feature; it is generic preprocessing.
        """
        y = np.asarray(y, dtype=np.float32)

        if y.size == 0:
            return y

        peak = np.max(np.abs(y))
        if peak > self.eps:
            y = y / peak

        return y

    def compute_stft(self, y: np.ndarray) -> SpectrogramResult:
        """
        Compute magnitude, power, and log-magnitude spectrogram.

        STFT/spectrogram computation is explicitly allowed.
        """
        X = librosa.stft(
            y,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=self.center,
        )

        magnitude = np.abs(X)
        power = magnitude ** 2

        # log1p keeps silence at 0 and compresses large values.
        log_magnitude = np.log1p(magnitude)

        frames = np.arange(magnitude.shape[1])
        times = librosa.frames_to_time(
            frames,
            sr=self.sr,
            hop_length=self.hop_length,
        )
        return SpectrogramResult(
            magnitude=magnitude,
            power=power,
            log_magnitude=log_magnitude,
            times=times,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )

    def mel_spectrogram(
        self,
        y: np.ndarray,
        n_mels: int = 80,
        fmin: float = 30.0,
        fmax: Optional[float] = None,
        use_power: bool = False,
        log_scale: float = 100.0,
    ) -> FilterbankResult:
        """
        Compute mel-filtered spectrogram manually from STFT.

        We avoid librosa.feature.melspectrogram to keep the operation explicit:
        STFT magnitude -> mel filter matrix -> matrix multiplication.
        """
        spec = self.compute_stft(y)

        if fmax is None:
            fmax = self.sr / 2

        mel_basis = librosa.filters.mel(
            sr=self.sr,
            n_fft=self.n_fft,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
        )

        source = spec.power if use_power else spec.magnitude
        filtered = mel_basis @ source

        # Course-style dynamic range compression:
        # log(1 + lambda * magnitude)
        log_filtered = np.log1p(log_scale * filtered)

        return FilterbankResult(
            filtered=filtered,
            log_filtered=log_filtered,
            times=spec.times,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            scale="mel",
        )

    def semitone_filterbank(
        self,
        fmin: float = 27.5,
        fmax: Optional[float] = None,
        bins_per_octave: int = 12,
    ) -> np.ndarray:
        """
        Build a simple triangular semitone filterbank.

        This is useful for LogFiltSpecFlux-style preprocessing.
        The filters are constructed manually, then applied by matrix product.
        """
        if fmax is None:
            fmax = self.sr / 2

        fft_freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft)

        # Center frequencies on equal-tempered semitone grid.
        centers = []
        f = fmin
        ratio = 2 ** (1 / bins_per_octave)

        while f <= fmax:
            centers.append(f)
            f *= ratio

        centers = np.asarray(centers)

        if len(centers) < 3:
            raise ValueError("Too few semitone bands. Check fmin/fmax.")

        fb = np.zeros((len(centers), len(fft_freqs)), dtype=np.float32)

        for i, center in enumerate(centers):
            left = center / ratio
            right = center * ratio

            left_slope = (fft_freqs - left) / (center - left)
            right_slope = (right - fft_freqs) / (right - center)

            triangle = np.maximum(0.0, np.minimum(left_slope, right_slope))

            # Area normalization avoids high-frequency bands dominating
            # only because they cover more FFT bins.
            area = np.sum(triangle)
            if area > self.eps:
                triangle = triangle / area

            fb[i] = triangle

        return fb

    def semitone_spectrogram(
        self,
        y: np.ndarray,
        fmin: float = 27.5,
        fmax: Optional[float] = None,
        log_scale: float = 100.0,
    ) -> FilterbankResult:
        """
        Compute a semitone-filtered log spectrogram manually.
        """
        spec = self.compute_stft(y)

        fb = self.semitone_filterbank(fmin=fmin, fmax=fmax)
        filtered = fb @ spec.magnitude
        log_filtered = np.log1p(log_scale * filtered)

        return FilterbankResult(
            filtered=filtered,
            log_filtered=log_filtered,
            times=spec.times,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            scale="semitone",
        )

    def get_representation(
        self,
        y: np.ndarray,
        representation: Literal["stft", "mel", "semitone"] = "mel",
    ) -> SpectrogramResult | FilterbankResult:
        """
        Convenience wrapper for experiments.
        """
        if representation == "stft":
            return self.compute_stft(y)

        if representation == "mel":
            return self.mel_spectrogram(y)

        if representation == "semitone":
            return self.semitone_spectrogram(y)

        raise ValueError(f"Unknown representation: {representation}")