"""EXP-016: train the final learned onset model on all 277 onset-labelled files
and save weights to models/onset_lr.npz.

This is the model used for the submission: trained on all available labelled
data (127 train + 150 extra), then applied to the unseen test set. The honest
generalization estimate is the 5-fold CV number from exp016_learned_onset.py
(onset F1 0.7729), NOT a re-evaluation on the training files.

The 02_learned_onset.ipynb notebook mirrors this script for reproducibility.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import config  # noqa: E402
from src.data_loader import DataLoader  # noqa: E402
from src.features import FeatureExtractor  # noqa: E402
from src import learned_onset as lo  # noqa: E402

CACHE = ROOT / "experiments" / ".cache"


def load_audio_cached(loader, stem, wav):
    cpath = CACHE / f"{stem}.npy"
    if cpath.exists():
        return np.load(cpath), config.audio.sample_rate
    y, sr = loader.load_audio(Path(wav))
    if y is not None:
        np.save(cpath, y)
    return y, sr


def main():
    # Feature config (matches the winning CV setting: ctx=5, label_w=1, w_pos=2)
    ctx, label_w, w_pos = 5, 1, 2.0
    odfs = ("superflux", "complex")
    config.onset.fusion_odfs = odfs

    loader = DataLoader()
    train = loader.load_train(ROOT / "data" / "processed" / "train")
    extra = loader.load_extra_onsets(ROOT / "data" / "processed" / "train_extra_onsets")
    all_data = {**train, **extra}
    print(f"Training on {len(all_data)} onset-labelled files (127 train + {len(extra)} extra)")

    fe = FeatureExtractor()
    fps = config.audio.sample_rate / config.audio.onset_hop_length

    channels_list, onsets_list = [], []
    t0 = time.time()
    for i, (stem, info) in enumerate(all_data.items()):
        y, sr = load_audio_cached(loader, stem, info["wav"])
        if y is None:
            continue
        channels_list.append(fe.onset_channels(y))
        onsets_list.append(np.asarray(info.get("onsets") or [], dtype=float))
        if (i + 1) % 50 == 0:
            print(f"  features {i+1}/{len(all_data)} ({time.time()-t0:.0f}s)")

    model = lo.fit(channels_list, onsets_list, fps, ctx=ctx, label_w=label_w,
                   w_pos=w_pos, odfs=odfs)
    out = ROOT / config.onset.learned_model_path
    model.save(out)
    print(f"Saved model to {out}")
    print(f"  ctx={model.ctx}, odfs={model.odfs}, feat_dim={model.w.shape[0]}, "
          f"learned_delta={config.onset.learned_delta}")


if __name__ == "__main__":
    main()