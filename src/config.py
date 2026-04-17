"""Configuration for the AMP Challenge pipeline."""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass
class AudioConfig:
    """Audio processing parameters."""
    sample_rate: int = 22050
    hop_length: int = 512
    onset_hop_length: int = 512
    beat_hop_length: int = 512
    tempo_hop_length: int = 1024
    onset_fft_size: int = 2048
    onset_fmin: float = 30.0
    onset_fmax: float = 17000.0


@dataclass
class OnsetConfig:
    """Onset detection parameters."""
    threshold: float = 0.08  # LOWER = more onsets (was 0.3)
    peak_distance: int = 2   # SMALLER = more peaks (was 3)
    smoothing_sigma: float = 1.0
    method: str = "superflux"  # or "flux"
    eval_window_ms: float = 50.0


@dataclass
class BeatConfig:
    """Beat tracking parameters."""
    tempo_min: float = 30.0   # LOWER to catch slow tempos (was 55)
    tempo_max: float = 200.0  # (was 215)
    tightness: float = 60.0
    eval_window_ms: float = 70.0


@dataclass
class TempoConfig:
    """Tempo estimation parameters."""
    eval_tolerance_percent: float = 8.0


@dataclass
class PathConfig:
    """Path configuration."""
    base_dir: Path = Path(__file__).parent.parent
    data_raw_dir: Path = Path(__file__).parent.parent / "data" / "raw"
    data_processed_dir: Path = Path(__file__).parent.parent / "data" / "processed"
    submissions_dir: Path = Path(__file__).parent.parent / "submissions"
    
    def __post_init__(self):
        """Create directories if they don't exist."""
        self.data_raw_dir.mkdir(parents=True, exist_ok=True)
        self.data_processed_dir.mkdir(parents=True, exist_ok=True)
        self.submissions_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class Config:
    """Main configuration."""
    audio: AudioConfig = None
    onset: OnsetConfig = None
    beat: BeatConfig = None
    tempo: TempoConfig = None
    paths: PathConfig = None
    
    def __post_init__(self):
        if self.audio is None:
            self.audio = AudioConfig()
        if self.onset is None:
            self.onset = OnsetConfig()
        if self.beat is None:
            self.beat = BeatConfig()
        if self.tempo is None:
            self.tempo = TempoConfig()
        if self.paths is None:
            self.paths = PathConfig()


# Singleton instance
config = Config()