"""Tempo strategy diagnostic: comb_fusion vs beat-activation AC vs ensembles."""
import sys
from pathlib import Path
import numpy as np
import mir_eval

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.config import config
from src.data_loader import DataLoader
from src.detectors import BeatTracker, TempoEstimator

CACHE = ROOT / "experiments" / ".cache"

def load_cached(loader, stem, wav):
    cp = CACHE / f"{stem}.npy"
    if cp.exists():
        return np.load(cp), config.audio.sample_rate
    y, sr = loader.load_audio(Path(wav)); 
    if y is not None: np.save(cp, y)
    return y, sr

def pair_from_primary(T):
    lo, hi = config.beat.tempo_min, config.beat.tempo_max
    if T * 2 <= hi: return [T, 2*T]
    if T / 2 >= lo: return [T/2, T]
    return [T]

def pscore(gt, est):
    if len(gt) == 1: ref, w = np.array([gt[0]/2, gt[0]]), 0.5
    elif len(gt) == 2: ref, w = np.array(gt), 0.5
    else: ref, w = np.array(gt[:2]), gt[2]
    e = np.array(est[:2]) if len(est) >= 2 else np.array([est[0], est[0]])
    return float(mir_eval.tempo.detection(ref, w, e, tol=0.08)[0])

loader = DataLoader()
train = loader.load_train(ROOT / "data" / "processed" / "train")
bt = BeatTracker(); te = TempoEstimator()
hop = config.audio.beat_hop_length

strats = {"A_combfusion": [], "B_beatAC": [], "C_two_indep": [], "D_agree_else_cf": []}
for stem, gt in train.items():
    if not gt.get("tempo"): continue
    y, sr = load_cached(loader, stem, gt["wav"])
    if y is None: continue
    te.estimate(y, sr); T_cf = te._primary
    env = bt._beat_odf(y, sr, hop); fps = sr / hop
    T_beat = bt._estimate_tempo_from_env(env, fps)
    A = pair_from_primary(T_cf)
    B = pair_from_primary(T_beat)
    C = sorted([T_cf, T_beat])                      # two independent estimates
    # D: if agree within 8%, use comb_fusion pair; else offer both primaries
    D = A if abs(T_cf - T_beat) / max(T_cf,1e-9) < 0.08 else sorted([T_cf, T_beat])
    strats["A_combfusion"].append(pscore(gt["tempo"], A))
    strats["B_beatAC"].append(pscore(gt["tempo"], B))
    strats["C_two_indep"].append(pscore(gt["tempo"], C))
    strats["D_agree_else_cf"].append(pscore(gt["tempo"], D))

print(f"tempo p-score on {len(strats['A_combfusion'])} files:")
for k, v in strats.items():
    print(f"  {k:18s} {np.mean(v):.4f}")
