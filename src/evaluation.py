"""Evaluation metrics for onset, beat, and tempo detection."""

import numpy as np
import mir_eval
from typing import Dict, List, Tuple, Optional


class Evaluator:
    """Evaluate predictions against ground truth."""
    
    @staticmethod
    def onset_fmeasure(reference: List[float], estimated: List[float], window: float = 0.05) -> float:
        """Compute F-measure for onset detection."""
        if not reference or not estimated:
            return 0.0
        
        # mir_eval uses 'window' parameter for onset
        try:
            f, _, _ = mir_eval.onset.f_measure(
                np.array(reference),
                np.array(estimated),
                window=window
            )
        except TypeError:
            # Fallback for older versions
            f, _, _ = mir_eval.onset.f_measure(
                np.array(reference),
                np.array(estimated)
            )
        return f
    
    @staticmethod
    def beat_fmeasure(reference: List[float], estimated: List[float], window: float = 0.07) -> float:
        """Compute F-measure for beat tracking."""
        if not reference or not estimated:
            return 0.0
        
        # mir_eval.beat.f_measure uses different parameter names
        try:
            # Try with correct parameter names for mir_eval
            f, _, _ = mir_eval.beat.f_measure(
                np.array(reference),
                np.array(estimated),
                tolerance=window
            )
        except TypeError:
            try:
                # Try without tolerance
                f, _, _ = mir_eval.beat.f_measure(
                    np.array(reference),
                    np.array(estimated)
                )
            except:
                return 0.0
        return f
    
    @staticmethod
    def tempo_score(reference: List[float], estimated: List[float], tolerance: float = 0.08) -> float:
        """
        Compute tempo score (p-score).
        
        For ground truth with two tempos, the score is weighted.
        For single tempo, returns 1 if within tolerance.
        """
        if not reference:
            return 0.0
        
        try:
            # Try to use mir_eval
            if len(reference) == 1:
                # Single tempo - check if within tolerance
                for est in estimated:
                    if abs(est - reference[0]) / reference[0] <= tolerance:
                        return 1.0
                return 0.0
            else:
                # Two tempos - use mir_eval if available
                try:
                    score = mir_eval.tempo.evaluate(reference, estimated)[0]
                    return score
                except:
                    # Simple check: is either estimated tempo within tolerance of either reference?
                    for ref in reference:
                        for est in estimated:
                            if abs(est - ref) / ref <= tolerance:
                                return 1.0
                    return 0.0
        except:
            return 0.0
    
    @staticmethod
    def evaluate_all(predictions: Dict, ground_truth: Dict, 
                     onset_window: float = 0.05,
                     beat_window: float = 0.07,
                     tempo_tolerance: float = 0.08) -> Dict[str, float]:
        """
        Evaluate all predictions against ground truth.
        
        Returns:
            Dict with 'onsets', 'beats', 'tempo', 'mean' scores
        """
        onset_scores = []
        beat_scores = []
        tempo_scores = []
        
        for stem, gt in ground_truth.items():
            if stem not in predictions:
                continue
            
            pred = predictions[stem]
            
            # Onsets
            if gt.get('onsets') and pred.get('onsets'):
                score = Evaluator.onset_fmeasure(gt['onsets'], pred['onsets'], onset_window)
                onset_scores.append(score)
            
            # Beats
            if gt.get('beats') and pred.get('beats'):
                score = Evaluator.beat_fmeasure(gt['beats'], pred['beats'], beat_window)
                beat_scores.append(score)
            
            # Tempo
            if gt.get('tempo') and pred.get('tempo'):
                score = Evaluator.tempo_score(gt['tempo'], pred['tempo'], tempo_tolerance)
                tempo_scores.append(score)
        
        scores = {
            'onsets': np.mean(onset_scores) if onset_scores else 0.0,
            'beats': np.mean(beat_scores) if beat_scores else 0.0,
            'tempo': np.mean(tempo_scores) if tempo_scores else 0.0,
        }
        scores['mean'] = np.mean([scores['onsets'], scores['beats'], scores['tempo']])
        
        return scores