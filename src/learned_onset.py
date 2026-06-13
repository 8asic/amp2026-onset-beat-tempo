"""EXP-016: pure-numpy learned onset activation.

A logistic-regression classifier (hand-written forward + gradient, numpy only)
predicts a per-frame onset probability from the existing ODF channels
(superflux bands + complex-domain, EXP-015). The probability is an *activation*
that feeds the unchanged OnsetDetector peak picker — the musical decision (what
is a peak) stays our own code, so this is challenge-rules compliant: no library
classifier, peak picker, or onset detector is used.

Held-out performance (5-fold CV over the 277 onset-labelled files):
onset F1 0.7729 vs EXP-015 fusion 0.7615 (+0.011), robust across peak-pick
delta in [0.10, 0.20]. Train/eval discipline: the classifier is scored only on
files it did not train on; the submission model is trained on all 277 and
applied to the unseen test set.

The feature builder here is the single source of truth — OnsetDetector imports
`frame_features` for inference so training and inference features are identical.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------- features ----
def stack_context(channels: np.ndarray, ctx: int) -> np.ndarray:
    """(n_chan, T) ODF channels -> (T, n_chan*(2*ctx+1)) context-stacked features.

    Each output frame sees its own ODF values plus +-ctx neighbouring frames
    (zero-padded at the edges), giving the classifier the local temporal shape
    the peak picker also relies on.
    """
    n_chan, T = channels.shape
    feats = []
    for d in range(-ctx, ctx + 1):
        shifted = np.zeros_like(channels)
        if d < 0:
            shifted[:, -d:] = channels[:, :T + d]
        elif d > 0:
            shifted[:, :T - d] = channels[:, d:]
        else:
            shifted = channels
        feats.append(shifted)
    return np.concatenate(feats, axis=0).T


def frame_features(channels: np.ndarray, ctx: int) -> np.ndarray:
    """Public alias: context-stacked per-frame features from ODF channels."""
    return stack_context(channels, ctx)


def make_labels(onsets, T: int, fps: float, label_w: int) -> np.ndarray:
    """Binary per-frame labels: 1 within +-label_w frames of a GT onset."""
    lab = np.zeros(T, dtype=np.float64)
    if onsets is None or len(onsets) == 0:
        return lab
    frames = np.round(np.asarray(onsets, dtype=float) * fps).astype(int)
    for f in frames:
        lo, hi = max(0, f - label_w), min(T, f + label_w + 1)
        lab[lo:hi] = 1.0
    return lab


# --------------------------------------------------------------- model ----
def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def train_lr(X: np.ndarray, y: np.ndarray, w_pos: float = 2.0,
             epochs: int = 200, lr: float = 0.5, l2: float = 1e-4):
    """Class-weighted logistic regression via full-batch gradient descent.

    Returns (w, b). Positive frames are weighted by `w_pos` to counter the
    onset/non-onset class imbalance.
    """
    n, d = X.shape
    w = np.zeros(d); b = 0.0
    sw = np.where(y > 0, w_pos, 1.0)
    swsum = sw.sum()
    for _ in range(epochs):
        p = _sigmoid(X @ w + b)
        g = (p - y) * sw
        gw = X.T @ g / swsum + l2 * w
        gb = g.sum() / swsum
        w -= lr * gw
        b -= lr * gb
    return w, b


@dataclass
class LearnedOnsetModel:
    """Trained logistic-regression onset activation model (self-describing).

    Stores feature standardisation (mu, sd), weights (w, b), the context width
    and the ODF channel names it was trained on, so inference reproduces the
    exact training feature space.
    """
    mu: np.ndarray
    sd: np.ndarray
    w: np.ndarray
    b: float
    ctx: int
    odfs: tuple

    def predict_activation(self, channels: np.ndarray) -> np.ndarray:
        """ODF channels (n_chan, T) -> per-frame onset activation (T,)."""
        X = stack_context(channels, self.ctx)
        Xn = (X - self.mu) / self.sd
        return _sigmoid(Xn @ self.w + self.b)

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path, mu=self.mu, sd=self.sd, w=self.w, b=np.float64(self.b),
            ctx=np.int64(self.ctx), odfs=np.array(list(self.odfs), dtype=object),
        )

    @classmethod
    def load(cls, path) -> "LearnedOnsetModel":
        d = np.load(path, allow_pickle=True)
        return cls(
            mu=d["mu"], sd=d["sd"], w=d["w"], b=float(d["b"]),
            ctx=int(d["ctx"]), odfs=tuple(d["odfs"].tolist()),
        )


def fit(channels_list, onsets_list, fps: float, ctx: int = 5, label_w: int = 1,
        w_pos: float = 2.0, odfs: tuple = ("superflux", "complex"),
        epochs: int = 200, lr: float = 0.5, l2: float = 1e-4) -> LearnedOnsetModel:
    """Train a LearnedOnsetModel from per-file ODF channels + GT onsets.

    channels_list: list of (n_chan, T) ODF arrays (from FeatureExtractor.onset_channels)
    onsets_list:   list of GT onset-time arrays (seconds), aligned with channels_list
    """
    Xs, ys = [], []
    for chans, onsets in zip(channels_list, onsets_list):
        Xs.append(stack_context(chans, ctx))
        ys.append(make_labels(onsets, chans.shape[1], fps, label_w))
    X = np.concatenate(Xs).astype(np.float64)
    y = np.concatenate(ys).astype(np.float64)
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-8
    Xn = (X - mu) / sd
    w, b = train_lr(Xn, y, w_pos=w_pos, epochs=epochs, lr=lr, l2=l2)
    return LearnedOnsetModel(mu=mu, sd=sd, w=w, b=float(b), ctx=ctx, odfs=tuple(odfs))