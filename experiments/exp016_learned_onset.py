"""EXP-016 prototype: pure-numpy LEARNED onset activation, k-fold held-out eval.

Goal: learn a per-frame onset activation from the existing ODF channels
(superflux bands + complex-domain) feeding the SAME hand-written peak picker,
and check whether it beats the EXP-015 fusion baseline (277-file onset 0.7615)
on HELD-OUT folds — not on training data.

Rules compliance: model is a pure-numpy logistic-regression / MLP (hand-written
forward + gradient). It outputs an activation; the musical decision (peak
picking) stays OnsetDetector._pick. No sklearn, torch, etc.

Honesty: 5-fold CV over the 277 onset-labelled files. The classifier trains on
frames from 4 folds and is scored (file-level onset F1 after peak picking) on
the held-out fold. The peak-pick delta is swept GLOBALLY and the best held-out
mean reported — this is a mild optimism (delta chosen on held-out); if the model
is going to win it must win clearly here first. A clean keep would re-confirm
with nested delta selection.

Usage:
    python experiments/exp016_learned_onset.py                 # logistic regression
    python experiments/exp016_learned_onset.py --model mlp --hidden 32
    python experiments/exp016_learned_onset.py --context 5 --label-w 2
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import mir_eval

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import config  # noqa: E402
from src.data_loader import DataLoader  # noqa: E402
from src.features import FeatureExtractor  # noqa: E402
from src.detectors import OnsetDetector  # noqa: E402

CACHE = ROOT / "experiments" / ".cache"
FEAT_CACHE = ROOT / "experiments" / ".cache_feat"
FEAT_CACHE.mkdir(parents=True, exist_ok=True)

FPS = config.audio.sample_rate / config.audio.onset_hop_length


def load_audio_cached(loader, stem, wav):
    cpath = CACHE / f"{stem}.npy"
    if cpath.exists():
        return np.load(cpath), config.audio.sample_rate
    y, sr = loader.load_audio(Path(wav))
    if y is not None:
        np.save(cpath, y)
    return y, sr


def build_channels(fe, y):
    """ODF channels (n_chan, T): superflux bands + complex-domain (EXP-015 set)."""
    return fe.onset_channels(y)  # uses config.onset.fusion_odfs


def stack_context(chans, ctx):
    """(n_chan, T) -> (T, n_chan*(2*ctx+1)) by stacking +-ctx frame neighbours."""
    n_chan, T = chans.shape
    feats = []
    for d in range(-ctx, ctx + 1):
        shifted = np.zeros_like(chans)
        if d < 0:
            shifted[:, -d:] = chans[:, :T + d]
        elif d > 0:
            shifted[:, :T - d] = chans[:, d:]
        else:
            shifted = chans
        feats.append(shifted)
    return np.concatenate(feats, axis=0).T  # (T, n_chan*(2ctx+1))


def make_labels(gt_onsets, T, label_w):
    lab = np.zeros(T, dtype=np.float64)
    if not gt_onsets:
        return lab
    frames = np.round(np.asarray(gt_onsets, dtype=float) * FPS).astype(int)
    for f in frames:
        lo, hi = max(0, f - label_w), min(T, f + label_w + 1)
        lab[lo:hi] = 1.0
    return lab


def build_dataset(loader, data, fe, ctx, label_w, tag):
    """Per-file (X, y, gt_onsets). Cached to disk keyed by feature config."""
    key = f"{tag}_ctx{ctx}_lw{label_w}_nb{config.onset.n_bands}_odfs{'-'.join(config.onset.fusion_odfs)}"
    cpath = FEAT_CACHE / f"{key}.npz"
    if cpath.exists():
        d = np.load(cpath, allow_pickle=True)
        return d["files"].tolist()
    files = []
    t0 = time.time()
    for i, (stem, info) in enumerate(data.items()):
        y, sr = load_audio_cached(loader, stem, info["wav"])
        if y is None:
            continue
        chans = build_channels(fe, y)
        X = stack_context(chans, ctx)
        lab = make_labels(info.get("onsets"), X.shape[0], label_w)
        files.append({"stem": stem, "X": X.astype(np.float32),
                      "y": lab.astype(np.float32),
                      "onsets": np.asarray(info.get("onsets") or [], dtype=float)})
        if (i + 1) % 50 == 0:
            print(f"  features {i+1}/{len(data)} ({time.time()-t0:.0f}s)")
    np.savez(cpath, files=np.array(files, dtype=object))
    print(f"  built {len(files)} files in {time.time()-t0:.0f}s -> cached")
    return files


# ---- pure-numpy models ----
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def train_lr(X, y, w_pos, epochs=200, lr=0.5, l2=1e-4):
    n, d = X.shape
    w = np.zeros(d); b = 0.0
    sw = np.where(y > 0, w_pos, 1.0)
    swsum = sw.sum()
    for _ in range(epochs):
        p = sigmoid(X @ w + b)
        g = (p - y) * sw
        gw = X.T @ g / swsum + l2 * w
        gb = g.sum() / swsum
        w -= lr * gw; b -= lr * gb
    return ("lr", w, b)


def train_mlp(X, y, w_pos, hidden=32, epochs=200, lr=0.3, l2=1e-4, seed=0):
    rng = np.random.default_rng(seed)
    n, d = X.shape
    W1 = rng.standard_normal((d, hidden)) * np.sqrt(2.0 / d)
    b1 = np.zeros(hidden)
    W2 = rng.standard_normal(hidden) * np.sqrt(1.0 / hidden)
    b2 = 0.0
    sw = np.where(y > 0, w_pos, 1.0)
    swsum = sw.sum()
    for _ in range(epochs):
        h_pre = X @ W1 + b1
        h = np.tanh(h_pre)
        p = sigmoid(h @ W2 + b2)
        g = (p - y) * sw                       # dL/dz2
        gW2 = h.T @ g / swsum + l2 * W2
        gb2 = g.sum() / swsum
        gh = np.outer(g, W2) * (1.0 - h ** 2)  # backprop through tanh
        gW1 = X.T @ gh / swsum + l2 * W1
        gb1 = gh.sum(axis=0) / swsum
        W2 -= lr * gW2; b2 -= lr * gb2
        W1 -= lr * gW1; b1 -= lr * gb1
    return ("mlp", W1, b1, W2, b2)


def predict(model, X):
    if model[0] == "lr":
        _, w, b = model
        return sigmoid(X @ w + b)
    _, W1, b1, W2, b2 = model
    return sigmoid(np.tanh(X @ W1 + b1) @ W2 + b2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lr", choices=["lr", "mlp"])
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--context", type=int, default=5, help="+-frames of context")
    ap.add_argument("--label-w", type=int, default=2, help="+-frames labelled positive")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--w-pos", type=float, default=5.0, help="positive class weight")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--odfs", default=None, help="EXP-019: comma ODF channels e.g. superflux,complex,hfc,phase")
    ap.add_argument("--n-bands", type=int, default=None, help="EXP-019: superflux mel-band count (more = finer spectral features)")
    args = ap.parse_args()

    if args.odfs is not None:
        config.onset.fusion_odfs = tuple(args.odfs.split(","))
    if args.n_bands is not None:
        config.onset.n_bands = args.n_bands

    loader = DataLoader()
    train = loader.load_train(ROOT / "data" / "processed" / "train")
    extra = loader.load_extra_onsets(ROOT / "data" / "processed" / "train_extra_onsets")
    all_data = {**train, **extra}
    print(f"Onset-labelled files: {len(all_data)} (127 train + {len(extra)} extra)")
    print(f"ODF channels: {config.onset.fusion_odfs}, n_bands={config.onset.n_bands}")

    fe = FeatureExtractor()
    print("Building features (cached)...")
    files = build_dataset(loader, all_data, fe, args.context, args.label_w, "all277")
    train_stems = set(train.keys())  # 127-corpus (resembles the test set)
    for f in files:
        f["subset"] = "c127" if f["stem"] in train_stems else "extra"
    n_feat = files[0]["X"].shape[1]
    print(f"feature dim = {n_feat}, model = {args.model}"
          + (f" hidden={args.hidden}" if args.model == "mlp" else ""))

    # k-fold split over files
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(files))
    folds = np.array_split(order, args.folds)

    od = OnsetDetector()
    deltas = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    # accumulate per-delta held-out F1 across all folds
    delta_scores = {d: [] for d in deltas}
    # at a fixed a-priori delta, split held-out F1 by subset (test resembles c127)
    FIXED_DELTA = 0.18
    sub_scores = {"c127": [], "extra": []}

    lr_default = 0.5 if args.model == "lr" else 0.3
    learn_rate = args.lr if args.lr is not None else lr_default

    for fi in range(args.folds):
        test_idx = set(folds[fi].tolist())
        tr = [files[i] for i in range(len(files)) if i not in test_idx]
        te = [files[i] for i in range(len(files)) if i in test_idx]

        Xtr = np.concatenate([f["X"] for f in tr]).astype(np.float64)
        ytr = np.concatenate([f["y"] for f in tr]).astype(np.float64)
        mu = Xtr.mean(axis=0); sd = Xtr.std(axis=0) + 1e-8
        Xtr = (Xtr - mu) / sd

        if args.model == "lr":
            model = train_lr(Xtr, ytr, args.w_pos, args.epochs, learn_rate)
        else:
            model = train_mlp(Xtr, ytr, args.w_pos, args.hidden, args.epochs,
                              learn_rate, seed=args.seed)

        for f in te:
            Xte = (f["X"].astype(np.float64) - mu) / sd
            act = predict(model, Xte)
            if f["onsets"].size == 0:
                continue
            for d in deltas:
                peaks = od._pick(act, FPS, d)
                est = np.array(peaks, dtype=int) / FPS
                if len(est):
                    fm, _, _ = mir_eval.onset.f_measure(f["onsets"], est, window=0.05)
                else:
                    fm = 0.0
                delta_scores[d].append(fm)
            # subset split at the fixed a-priori delta
            peaks = od._pick(act, FPS, FIXED_DELTA)
            est = np.array(peaks, dtype=int) / FPS
            fm = mir_eval.onset.f_measure(f["onsets"], est, window=0.05)[0] if len(est) else 0.0
            sub_scores[f["subset"]].append(fm)
        print(f"  fold {fi+1}/{args.folds} done ({len(te)} files)")

    print("\nHeld-out onset F1 by peak-pick delta (277-file 5-fold CV):")
    best_d, best_f = None, -1.0
    for d in deltas:
        m = float(np.mean(delta_scores[d]))
        flag = ""
        if m > best_f:
            best_f, best_d = m, d
        print(f"  delta={d:.2f}: F1={m:.4f}  ({len(delta_scores[d])} files)")
    print(f"\nBEST held-out: delta={best_d:.2f}  F1={best_f:.4f}")
    print(f"EXP-015 fusion baseline (277): 0.7615")
    print(f"Verdict: {'BEATS baseline' if best_f > 0.7615 else 'does NOT beat baseline'} "
          f"({best_f - 0.7615:+.4f})")

    print(f"\nHeld-out by subset (fixed delta={FIXED_DELTA}) — test set resembles c127:")
    c127 = float(np.mean(sub_scores["c127"])) if sub_scores["c127"] else 0.0
    extra = float(np.mean(sub_scores["extra"])) if sub_scores["extra"] else 0.0
    print(f"  LEARNED  c127={c127:.4f} ({len(sub_scores['c127'])})  "
          f"extra={extra:.4f} ({len(sub_scores['extra'])})")
    print(f"  FUSION   c127=0.8055              extra=0.7242  (non-learned, full-set scores)")
    print(f"  -> on the test-relevant c127 corpus, learned {'beats' if c127 > 0.8055 else 'trails'} "
          f"fusion ({c127 - 0.8055:+.4f})")


if __name__ == "__main__":
    main()