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
        Detect onsets in audio.
        
        Args:
            y: Audio signal
            sr: Sample rate
        
        Returns:
            Onset times in seconds
        """
        # Compute onset strength
        strength = self.fe.onset_strength(
            y, 
            method=self.cfg.onset.method
        )
        
        # Adaptive threshold
        threshold = self._adaptive_threshold(strength)
        
        # Find peaks
        peaks = find_peaks(
            strength,
            height=threshold,
            distance=self.cfg.onset.peak_distance
        )[0]
        
        # Convert frames to time
        onset_times = librosa.frames_to_time(
            peaks,
            sr=sr,
            hop_length=self.cfg.audio.onset_hop_length
        )
        
        return onset_times
    
    def _adaptive_threshold(self, strength: np.ndarray) -> float:
        """Compute adaptive threshold based on moving mean."""
        window = int(0.5 * self.cfg.audio.sample_rate / self.cfg.audio.onset_hop_length)
        if window % 2 == 0:
            window += 1
        
        # Simple moving mean
        kernel = np.ones(window) / window
        moving_mean = np.convolve(strength, kernel, mode='same')
        
        return moving_mean + self.cfg.onset.threshold


class BeatTracker:
    """Track beats in audio."""
    
    def __init__(self, cfg=None):
        self.cfg = cfg or config
    
    def track(self, y: np.ndarray, sr: int, tempo: Optional[float] = None) -> Tuple[np.ndarray, float]:
        """
        Track beats in audio.
        
        Args:
            y: Audio signal
            sr: Sample rate
            tempo: Optional tempo estimate (auto-detected if None)
        
        Returns:
            (beat_times, estimated_tempo)
        """
        # Compute onset envelope
        onset_env = librosa.onset.onset_strength(
            y=y,
            sr=sr,
            hop_length=self.cfg.audio.beat_hop_length
        )
        
        # Estimate tempo if not provided
        if tempo is None:
            # Use librosa's tempo function (different parameter names in older versions)
            try:
                # Try newer librosa (>=0.10)
                tempo = librosa.beat.tempo(
                    onset_envelope=onset_env,
                    sr=sr,
                    hop_length=self.cfg.audio.beat_hop_length,
                    start_bpm=self.cfg.beat.tempo_min,
                    end_bpm=self.cfg.beat.tempo_max
                )[0]
            except TypeError:
                # Fallback for older librosa
                tempo = librosa.beat.tempo(
                    onset_envelope=onset_env,
                    sr=sr,
                    hop_length=self.cfg.audio.beat_hop_length,
                    start_bpm=self.cfg.beat.tempo_min
                )[0]
            
            if isinstance(tempo, np.ndarray):
                tempo = float(tempo[0]) if len(tempo) > 0 else 120.0
        
        # Detect beats
        try:
            # Try newer librosa
            beat_frames = librosa.beat.beat_track(
                onset_envelope=onset_env,
                sr=sr,
                hop_length=self.cfg.audio.beat_hop_length,
                start_bpm=tempo,
                tightness=self.cfg.beat.tightness,
                trim=False
            )[1]
        except TypeError:
            # Fallback for older librosa
            beat_frames = librosa.beat.beat_track(
                onset_envelope=onset_env,
                sr=sr,
                hop_length=self.cfg.audio.beat_hop_length,
                start_bpm=tempo,
                tightness=self.cfg.beat.tightness
            )[1]
        
        # Convert to time
        beat_times = librosa.frames_to_time(
            beat_frames,
            sr=sr,
            hop_length=self.cfg.audio.beat_hop_length
        )
        
        return beat_times, float(tempo)


class TempoEstimator:
    """Estimate tempo from audio."""
    
    def __init__(self, cfg=None):
        self.cfg = cfg or config
    
    def estimate(self, y: np.ndarray, sr: int) -> List[float]:
        """
        Estimate tempo(s) from audio.
        
        Returns:
            List of 1-2 tempo estimates (primary, optional secondary)
        """
        # Compute onset envelope
        onset_env = librosa.onset.onset_strength(
            y=y,
            sr=sr,
            hop_length=self.cfg.audio.tempo_hop_length
        )
        
        # Estimate tempo using librosa (handle different parameter names)
        try:
            # Try newer librosa (>=0.10) with end_bpm
            tempo_fft = librosa.beat.tempo(
                onset_envelope=onset_env,
                sr=sr,
                hop_length=self.cfg.audio.tempo_hop_length,
                start_bpm=self.cfg.beat.tempo_min,
                end_bpm=self.cfg.beat.tempo_max
            )[0]
        except TypeError:
            # Fallback for older librosa without end_bpm
            tempo_fft = librosa.beat.tempo(
                onset_envelope=onset_env,
                sr=sr,
                hop_length=self.cfg.audio.tempo_hop_length,
                start_bpm=self.cfg.beat.tempo_min
            )[0]
        
        tempos = [float(tempo_fft)]
        
        # Check for half/double tempo as secondary
        if tempo_fft * 2 <= self.cfg.beat.tempo_max:
            tempos.append(float(tempo_fft * 2))
        elif tempo_fft / 2 >= self.cfg.beat.tempo_min:
            tempos.append(float(tempo_fft / 2))
        
        return tempos


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