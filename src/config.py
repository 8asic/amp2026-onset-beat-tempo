"""Configuration for the AMP Challenge pipeline."""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass
class AudioConfig:
    """Audio processing parameters."""
    sample_rate: int = 22050
    hop_length: int = 512
    onset_hop_length: int = 256
    beat_hop_length: int = 512
    tempo_hop_length: int = 1024
    onset_fft_size: int = 2048
    onset_fmin: float = 30.0
    onset_fmax: float = 11000.0  # keep below Nyquist (sr/2 = 11025 for sr=22050)
    onset_n_mels: int = 82
    superflux_gamma: float = 200.0
    superflux_mu: int = 3
    whiten: bool = True            # EXP-011: adaptive spectral whitening
    whiten_decay: float = 0.995    # causal running-peak decay per frame
    whiten_floor: float = 0.10     # per-bin floor as fraction of bin max
    hpss: bool = False             # EXP-014: REJECTED — hurts generalization (see log)
    hpss_kernel_harm: int = 127    # median filter length along time axis (harmonic)
    hpss_kernel_perc: int = 17     # median filter length along freq axis (percussive)


@dataclass
class OnsetConfig:
    """Onset detection parameters."""
    threshold: float = 0.055  # EXP-015 fusion (superflux+complex) optimum on 277-file set
    peak_distance: int = 2   # SMALLER = more peaks (was 3)
    smoothing_sigma: float = 1.0
    method: str = "superflux"  # or "flux"
    eval_window_ms: float = 50.0
    multiband: bool = True    # EXP-010: per-band superflux peak picking (fallback when fusion off)
    n_bands: int = 2          # mel-band groups for multiband picking
    merge_tol_ms: float = 15.0  # merge cross-band peaks within this tolerance
    fusion: bool = True       # EXP-015: multi-ODF channel peak-picking + merge
    fusion_odfs: tuple = ("superflux", "complex")  # channels to fuse (hfc/phase rejected)
    learned: bool = False     # EXP-016: pure-numpy learned onset activation -> picker
    learned_model_path: str = "models/onset_lr.npz"  # trained LR weights
    learned_delta: float = 0.18  # peak-pick threshold on the learned activation


@dataclass
class BeatConfig:
    """Beat tracking parameters."""
    tempo_min: float = 30.0
    tempo_max: float = 200.0
    tempo_search_min: float = 60.0    # AC search floor; avoids measure-level false peaks
    tempo_method: str = "comb_fusion"  # "argmax" | "comb" | "comb_fusion" (EXP-008)
    tempo_comb_harmonics: int = 2     # # integer harmonics summed in comb salience
    dp_alpha: float = 100.0          # Ellis DP tightness (kept for reference)
    dp_transition_width: float = 0.10  # tight window: ±fraction of beat period
    dp_transition_lambda: float = 1.0  # Gaussian penalty strength
    beat_two_pass: bool = False        # EXP-009: no-op on this data (comb_fusion lags already accurate)
    beat_octave_select: bool = True    # EXP-012: pick beat octave by on/off-beat onset contrast (slow tempos only)
    beat_octave_gate: float = 78.0     # only octave-select when base tempo below this
    eval_window_ms: float = 70.0
    learned: bool = False              # EXP-018: BLSTM beat activation -> tempo+DP (flip to True when weights land)
    learned_model_path: str = "models/beat_blstm.pt"  # PyTorch BLSTM weights (Colab-trained)
    learned_decode: str = "B"          # "B": comb_fusion tempo + octave-select (default); "A": _estimate_tempo_from_env


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