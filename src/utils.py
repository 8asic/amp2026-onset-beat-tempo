"""Utility functions for file handling and extraction."""

import json
import subprocess
import zipfile
import shutil
from datetime import datetime
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


def get_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except Exception:
        return "no-git"


def save_versioned_submission(
    predictions: dict,
    submissions_dir,
    experiment_id: str,
    val_scores: dict = None,
    notes: str = "",
):
    submissions_dir = Path(submissions_dir)
    commit = get_git_commit()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = submissions_dir / f"{experiment_id}_{timestamp}_{commit}"
    run_dir.mkdir(parents=True, exist_ok=True)

    pred_path = run_dir / "predictions.json"
    with open(pred_path, "w") as f:
        json.dump(predictions, f, indent=2)

    metadata = {
        "experiment_id": experiment_id,
        "timestamp": timestamp,
        "git_commit": commit,
        "val_scores": val_scores or {},
        "notes": notes,
        "n_test_files": len(predictions),
    }
    with open(run_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved: {pred_path}")
    return pred_path