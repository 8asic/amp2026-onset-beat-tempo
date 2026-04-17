"""Utility functions for file handling and extraction."""

import zipfile
import shutil
from pathlib import Path
from typing import List, Optional


def extract_if_needed(zip_path: Path, dest_dir: Path, marker: str = ".extracted") -> bool:
    """
    Extract zip file only if not already extracted.
    Handles zips that contain a single root folder by moving contents up.
    """
    marker_path = dest_dir / marker
    
    if marker_path.exists():
        return False
    
    # Clear existing directory if it exists
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(dest_dir)
    
    # If the zip extracted into a single subfolder, move contents up
    items = list(dest_dir.iterdir())
    if len(items) == 1 and items[0].is_dir():
        subfolder = items[0]
        for item in subfolder.iterdir():
            shutil.move(str(item), str(dest_dir / item.name))
        subfolder.rmdir()
        print(f"  Flattened: removed subfolder '{subfolder.name}'")
    
    # Also handle case where there's a folder with same name as dest_dir
    for item in list(dest_dir.iterdir()):
        if item.is_dir() and item.name == dest_dir.name:
            for subitem in item.iterdir():
                shutil.move(str(subitem), str(dest_dir / subitem.name))
            item.rmdir()
            print(f"  Flattened: removed nested '{item.name}' folder")
    
    marker_path.touch()
    return True


def load_onsets_gt(path: Path) -> List[float]:
    """
    Load onset ground truth from file.
    Handles both single-value-per-line and tab-separated formats.
    """
    if not path.exists():
        return []
    
    onsets = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Handle tab-separated format: "time\tvalue"
            parts = line.split()
            if parts:
                try:
                    # Take the first part as the time
                    onsets.append(float(parts[0]))
                except ValueError:
                    continue
    
    return onsets


def load_beats_gt(path: Path) -> List[float]:
    """Load beat ground truth from file."""
    return load_onsets_gt(path)


def load_tempo_gt(path: Path) -> List[float]:
    """
    Load tempo ground truth from file.
    Returns list of tempos (may have 1 or 2 values).
    Handles tab-separated format if present.
    """
    if not path.exists():
        return []
    
    with open(path, 'r') as f:
        content = f.read().strip()
        if not content:
            return []
        
        # Split by whitespace (handles spaces, tabs, newlines)
        parts = list(map(float, content.split()))
        
        if len(parts) == 1:
            return [parts[0]]
        elif len(parts) == 2:
            # Two tempos without weight
            return [parts[0], parts[1]]
        elif len(parts) == 3:
            # Two tempos with weight
            return [parts[0], parts[1]]
        return []