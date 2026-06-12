"""Standalone 127-file validation harness for fast experiment iteration.

Usage:
    python experiments/validate.py                  # current config
    python experiments/validate.py --tempo-method argmax
    python experiments/validate.py --tempo-method comb_fusion --diag-tempo

Audio is cached to experiments/.cache/*.npy so reruns are fast.
mir_eval.onset.f_measure returns (f_measure, precision, recall) -- name the
first value `f` (see memory/gotchas.md).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import mir_eval

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import config  # noqa: E402
from src.data_loader import DataLoader  # noqa: E402
from src.detectors import Pipeline  # noqa: E402

CACHE = ROOT / "experiments" / ".cache"
CACHE.mkdir(parents=True, exist_ok=True)


def load_audio_cached(loader: DataLoader, stem: str, wav: str):
    cpath = CACHE / f"{stem}.npy"
    if cpath.exists():
        return np.load(cpath), config.audio.sample_rate
    y, sr = loader.load_audio(Path(wav))
    if y is not None:
        np.save(cpath, y)
    return y, sr


def tempo_pscore(gt_tempo, est_tempo):
    if len(gt_tempo) == 1:
        ref_tempi = np.array([gt_tempo[0] / 2, gt_tempo[0]])
        ref_weight = 0.5
    elif len(gt_tempo) == 2:
        ref_tempi = np.array(gt_tempo)
        ref_weight = 0.5
    else:
        ref_tempi = np.array(gt_tempo[:2])
        ref_weight = gt_tempo[2]
    est_tempi = np.array(est_tempo[:2])
    p, _, _ = mir_eval.tempo.detection(ref_tempi, ref_weight, est_tempi, tol=0.08)
    return float(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tempo-method", default=None, help="argmax | comb | comb_fusion")
    ap.add_argument("--harmonics", type=int, default=None, help="comb harmonics")
    ap.add_argument("--diag-tempo", action="store_true", help="list wrong-tempo files")
    ap.add_argument("--multiband", action="store_true", help="enable multiband onset")
    ap.add_argument("--n-bands", type=int, default=None, help="multiband band count")
    ap.add_argument("--merge-tol", type=float, default=None, help="merge tol ms")
    ap.add_argument("--threshold", type=float, default=None, help="onset threshold")
    ap.add_argument("--whiten", action="store_true", help="enable adaptive whitening")
    ap.add_argument("--whiten-decay", type=float, default=None, help="whiten decay")
    ap.add_argument("--whiten-floor", type=float, default=None, help="whiten floor")
    args = ap.parse_args()

    if args.tempo_method is not None:
        config.beat.tempo_method = args.tempo_method
    if args.harmonics is not None:
        config.beat.tempo_comb_harmonics = args.harmonics
    if args.multiband:
        config.onset.multiband = True
    if args.n_bands is not None:
        config.onset.n_bands = args.n_bands
    if args.merge_tol is not None:
        config.onset.merge_tol_ms = args.merge_tol
    if args.threshold is not None:
        config.onset.threshold = args.threshold
    if args.whiten:
        config.audio.whiten = True
    if args.whiten_decay is not None:
        config.audio.whiten_decay = args.whiten_decay
    if args.whiten_floor is not None:
        config.audio.whiten_floor = args.whiten_floor

    loader = DataLoader()
    train_dir = ROOT / "data" / "processed" / "train"
    train = loader.load_train(train_dir)

    pipe = Pipeline()

    onset_f, beat_f, tempo_p = [], [], []
    tempo_wrong = []

    for stem, gt in train.items():
        y, sr = load_audio_cached(loader, stem, gt["wav"])
        if y is None:
            continue
        onsets, beats, tempos = pipe.process_file(y, sr)

        if gt.get("onsets") and len(onsets):
            f, _, _ = mir_eval.onset.f_measure(
                np.array(gt["onsets"]), np.array(onsets), window=0.05
            )
            onset_f.append(f)
        if gt.get("beats") and len(beats):
            beat_f.append(mir_eval.beat.f_measure(np.array(gt["beats"]), np.array(beats)))
        if gt.get("tempo") and tempos:
            p = tempo_pscore(gt["tempo"], tempos)
            tempo_p.append(p)
            if p < 0.5 and args.diag_tempo:
                tempo_wrong.append((stem, gt["tempo"], [round(t, 1) for t in tempos]))

    mo = np.mean(onset_f) if onset_f else 0.0
    mb = np.mean(beat_f) if beat_f else 0.0
    mt = np.mean(tempo_p) if tempo_p else 0.0
    mean = (mo + mb + mt) / 3

    print(f"tempo_method={config.beat.tempo_method}")
    print(f"  Onset F1:  {mo:.4f}  ({len(onset_f)} files)")
    print(f"  Beat F1:   {mb:.4f}  ({len(beat_f)} files)")
    print(f"  Tempo:     {mt:.4f}  ({len(tempo_p)} files)")
    print(f"  MEAN:      {mean:.4f}")

    if args.diag_tempo:
        print(f"\nWrong-tempo files (p<0.5): {len(tempo_wrong)}")
        for stem, gt_t, est_t in tempo_wrong:
            print(f"  {stem}: GT={gt_t} EST={est_t}")


if __name__ == "__main__":
    main()
