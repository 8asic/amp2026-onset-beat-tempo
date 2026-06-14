"""Local evaluation utilities for AMP challenge."""

from typing import Dict, List, Tuple

import numpy as np


def match_events(
    reference: np.ndarray,
    estimated: np.ndarray,
    tolerance: float,
) -> Tuple[int, int, int]:
    """
    Match estimated event times to reference event times.

    Args:
        reference: ground-truth times in seconds
        estimated: predicted times in seconds
        tolerance: matching tolerance in seconds

    Returns:
        tp, fp, fn
    """
    reference = np.asarray(reference, dtype=float)
    estimated = np.asarray(estimated, dtype=float)

    if len(reference) == 0 and len(estimated) == 0:
        return 0, 0, 0

    if len(reference) == 0:
        return 0, len(estimated), 0

    if len(estimated) == 0:
        return 0, 0, len(reference)

    ref_used = np.zeros(len(reference), dtype=bool)
    est_used = np.zeros(len(estimated), dtype=bool)

    # Greedy matching by closest pair within tolerance.
    pairs = []
    for i, est in enumerate(estimated):
        distances = np.abs(reference - est)
        j = int(np.argmin(distances))
        if distances[j] <= tolerance:
            pairs.append((distances[j], i, j))

    pairs.sort(key=lambda x: x[0])

    tp = 0
    for _, i, j in pairs:
        if not est_used[i] and not ref_used[j]:
            est_used[i] = True
            ref_used[j] = True
            tp += 1

    fp = int(np.sum(~est_used))
    fn = int(np.sum(~ref_used))

    return tp, fp, fn


def precision_recall_f1(tp: int, fp: int, fn: int) -> Dict[str, float]:
    """Compute precision, recall, and F1."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def evaluate_events(
    references: Dict[str, np.ndarray],
    estimates: Dict[str, np.ndarray],
    tolerance: float,
) -> Dict[str, float]:
    """
    Evaluate onset or beat predictions over many files.
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0

    per_file = {}

    for key, ref in references.items():
        est = estimates.get(key, np.array([]))

        tp, fp, fn = match_events(ref, est, tolerance)
        metrics = precision_recall_f1(tp, fp, fn)

        per_file[key] = metrics

        total_tp += tp
        total_fp += fp
        total_fn += fn

    overall = precision_recall_f1(total_tp, total_fp, total_fn)
    overall["per_file"] = per_file

    return overall


def parse_tempo_annotation(tempo_gt) -> Tuple[float, float, float]:
    """
    Parse challenge tempo annotation.

    Supported formats:
    - [100]
    - [60, 120]
    - [60, 120, 0.8]

    Returns:
        t_lo, t_hi, weight_for_lower
    """
    values = list(tempo_gt)

    if len(values) == 1:
        t = float(values[0])
        return t, t, 1.0

    if len(values) == 2:
        t_lo = float(values[0])
        t_hi = float(values[1])

        if t_lo > t_hi:
            t_lo, t_hi = t_hi, t_lo

        # If no annotator weight is provided, treat both tempi as equally valid.
        return t_lo, t_hi, 0.5

    if len(values) == 3:
        t_lo = float(values[0])
        t_hi = float(values[1])
        w = float(values[2])

        if t_lo > t_hi:
            t_lo, t_hi = t_hi, t_lo
            w = 1.0 - w

        return t_lo, t_hi, w

    raise ValueError(f"Invalid tempo annotation: {tempo_gt}")


def tempo_correct(predicted: float, reference: float, tolerance_percent: float = 8.0) -> bool:
    """
    Check whether predicted tempo is within relative tolerance.
    """
    if reference <= 0:
        return False

    rel_error = abs(predicted - reference) / reference
    return rel_error <= tolerance_percent / 100.0


def evaluate_tempo_file(
    reference_annotation,
    predicted_tempos: List[float],
    tolerance_percent: float = 8.0,
) -> float:
    """
    Compute challenge-style tempo p-score for one file.

    If the annotation has two tempi:
        score = w * correct_lower + (1 - w) * correct_higher

    If there is one tempo:
        score is 1 if any prediction matches it, otherwise 0.
    """
    t_lo, t_hi, w = parse_tempo_annotation(reference_annotation)

    predicted_tempos = [float(t) for t in predicted_tempos]

    correct_lo = any(
        tempo_correct(pred, t_lo, tolerance_percent)
        for pred in predicted_tempos
    )

    correct_hi = any(
        tempo_correct(pred, t_hi, tolerance_percent)
        for pred in predicted_tempos
    )

    if t_lo == t_hi:
        return 1.0 if correct_lo else 0.0

    return w * float(correct_lo) + (1.0 - w) * float(correct_hi)


def evaluate_tempo(
    references: Dict[str, List[float]],
    estimates: Dict[str, List[float]],
    tolerance_percent: float = 8.0,
) -> Dict[str, float]:
    """
    Evaluate tempo predictions over many files.
    """
    per_file = {}
    scores = []

    for key, ref in references.items():
        pred = estimates.get(key, [])
        score = evaluate_tempo_file(ref, pred, tolerance_percent)
        per_file[key] = score
        scores.append(score)

    mean_score = float(np.mean(scores)) if scores else 0.0

    return {
        "p_score": mean_score,
        "per_file": per_file,
    }