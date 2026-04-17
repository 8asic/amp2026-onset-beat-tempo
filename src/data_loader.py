"""Data loading for training and test datasets."""

import librosa
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from tqdm import tqdm

from .config import config
from .utils import load_onsets_gt, load_beats_gt, load_tempo_gt


class DataLoader:
    """Load and manage audio files with annotations."""
    
    def __init__(self, cfg=None):
        self.cfg = cfg or config
        self.sr = self.cfg.audio.sample_rate
    
    def load_audio(self, filepath: Path) -> Tuple[Optional[np.ndarray], int]:
        """
        Load audio file.
        
        Returns:
            (audio_signal, sample_rate) or (None, sr) on error
        """
        try:
            y, sr = librosa.load(filepath, sr=self.sr)
            return y, sr
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None, self.sr
    
    def load_train(self, data_dir: Path) -> Dict[str, Dict]:
        """
        Load training data from processed directory.
        
        Returns:
            Dict mapping file stem to {
                'wav': path,
                'onsets': List[float],
                'beats': List[float],
                'tempo': List[float]
            }
        """
        data = {}
        wav_files = list(data_dir.glob("*.wav"))
        
        for wav_path in tqdm(wav_files, desc="Loading training data"):
            stem = wav_path.stem
            
            data[stem] = {
                'wav': str(wav_path),
                'onsets': load_onsets_gt(data_dir / f"{stem}.onsets.gt"),
                'beats': load_beats_gt(data_dir / f"{stem}.beats.gt"),
                'tempo': load_tempo_gt(data_dir / f"{stem}.tempo.gt"),
            }
        
        return data
    
    def load_test(self, data_dir: Path) -> Dict[str, Dict]:
        """
        Load test data from processed directory.
        
        Returns:
            Dict mapping file stem to {'wav': path, 'onsets': [], 'beats': [], 'tempo': []}
        """
        data = {}
        wav_files = sorted(data_dir.glob("*.wav"))
        
        for wav_path in tqdm(wav_files, desc="Loading test data"):
            stem = wav_path.stem
            data[stem] = {
                'wav': str(wav_path),
                'onsets': [],
                'beats': [],
                'tempo': [],
            }
        
        return data
    
    def load_extra_onsets(self, data_dir: Path) -> Dict[str, Dict]:
        """Load extra onsets-only training data."""
        data = {}
        wav_files = list(data_dir.glob("*.wav"))
        
        for wav_path in tqdm(wav_files, desc="Loading extra onsets"):
            stem = wav_path.stem
            data[stem] = {
                'wav': str(wav_path),
                'onsets': load_onsets_gt(data_dir / f"{stem}.onsets.gt"),
                'beats': [],
                'tempo': [],
            }
        
        return data
    
    def load_extra_tempobeats(self, data_dir: Path) -> Dict[str, Dict]:
        """Load extra tempo+beats training data."""
        data = {}
        wav_files = list(data_dir.glob("*.wav"))
        
        for wav_path in tqdm(wav_files, desc="Loading extra tempo/beats"):
            stem = wav_path.stem
            data[stem] = {
                'wav': str(wav_path),
                'onsets': [],
                'beats': load_beats_gt(data_dir / f"{stem}.beats.gt"),
                'tempo': load_tempo_gt(data_dir / f"{stem}.tempo.gt"),
            }
        
        return data