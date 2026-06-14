"""Configuration for the AMP Challenge pipeline."""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass
class AudioConfig:
    """Audio processing parameters."""

    # Keep original challenge sampling rate.
    sample_rate: int = 44100

    # 10 ms hop at 44.1 kHz.
    hop_length: int = 441
    onset_hop_length: int = 441
    beat_hop_length: int = 441
    tempo_hop_length: int = 441

    # ~23 ms STFT window at 44.1 kHz.
    onset_fft_size: int = 1024

    onset_fmin: float = 30.0
    onset_fmax: float = 17000.0


@dataclass
class OnsetConfig:
    threshold: float = 0.06
    peak_distance: int = 2
    smoothing_sigma: float = 1.0
    method: str = "log_mel_flux"
    eval_window_ms: float = 50.0


@dataclass
class BeatConfig:
    tempo_min: float = 28.0
    tempo_max: float = 240.0
    tightness: float = 60.0
    eval_window_ms: float = 70.0
    beat_flux_weights: Tuple[float, float, float] = (0.45, 0.40, 0.15)

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