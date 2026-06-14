"""Manual onset, tempo, and beat detection algorithms for AMP challenge."""

from typing import List, Optional, Tuple

import numpy as np

from .config import config
from .features import FeatureExtractor, DetectionFunctionResult


class OnsetDetector:
    """Detect note onsets from a manually computed detection function."""

    def __init__(self, cfg=None):
        self.cfg = cfg or config
        self.fe = FeatureExtractor(self.cfg)

    def detect(self, y: np.ndarray, sr: int) -> np.ndarray:
        """
        Detect onset times in seconds.

        This uses:
        - manual detection function from FeatureExtractor
        - manual adaptive threshold
        - manual local-maximum peak picking
        """
        result = self.fe.onset_strength(y, method=self.cfg.onset.method)
        peaks = self.pick_peaks(
            result.values,
            threshold_offset=self.cfg.onset.threshold,
            min_distance=self.cfg.onset.peak_distance,
        )
        return result.times[peaks]

    def pick_peaks(
    self,
    d: np.ndarray,
    threshold_offset: float = 0.08,
    min_distance: int = 2,
    local_window: int = 3,
    adaptive_window_sec: float = 0.5,
    adaptive_lambda: float = 1.0,
) -> np.ndarray:
        """
        Manual peak picking using lecture-style adaptive thresholding.

        Lecture formula:
            threshold[t] = delta + lambda * median(d[t-M/2 : t+M/2])
        """
        d = np.asarray(d, dtype=np.float32)

        if d.size == 0:
            return np.array([], dtype=int)

        adaptive_window = int(
            adaptive_window_sec
            * self.cfg.audio.sample_rate
            / self.cfg.audio.onset_hop_length
        )
        adaptive_window = max(adaptive_window, 3)

        if adaptive_window % 2 == 0:
            adaptive_window += 1

        half_adapt = adaptive_window // 2
        local_median = np.zeros_like(d, dtype=np.float32)

        for t in range(len(d)):
            left = max(0, t - half_adapt)
            right = min(len(d), t + half_adapt + 1)
            local_median[t] = np.median(d[left:right])

        threshold = threshold_offset + adaptive_lambda * local_median

        half = local_window // 2
        candidate_peaks = []

        for i in range(half, len(d) - half):
            left = i - half
            right = i + half + 1
            neighborhood = d[left:right]

            is_local_max = d[i] == np.max(neighborhood)
            is_strict_enough = d[i] > d[i - 1] and d[i] >= d[i + 1]
            is_above_threshold = d[i] >= threshold[i]

            if is_local_max and is_strict_enough and is_above_threshold:
                candidate_peaks.append(i)

        if not candidate_peaks:
            return np.array([], dtype=int)

        selected = []

        for peak in candidate_peaks:
            if not selected:
                selected.append(peak)
                continue

            if peak - selected[-1] >= min_distance:
                selected.append(peak)
            else:
                if d[peak] > d[selected[-1]]:
                    selected[-1] = peak

        return np.asarray(selected, dtype=int)


class TempoEstimator:
    """Estimate tempo from a manually computed onset detection function."""

    def __init__(self, cfg=None):
        self.cfg = cfg or config
        self.fe = FeatureExtractor(self.cfg)

    def estimate(self, y: np.ndarray, sr: int) -> List[float]:
        """
        Estimate one or two tempo guesses using pair-based autocorrelation scoring.

        The challenge tempo annotations often have the form [T, 2T].
        Therefore, this method scores tempo pairs rather than only the strongest
        individual autocorrelation peak.
        """
        result = self.fe.onset_strength(y, method=self.cfg.onset.method)

        bpm_candidates, scores = self.autocorrelation_tempo_candidates(
            result,
            top_k=12,
        )

        if len(bpm_candidates) == 0:
            return [120.0]

        min_bpm = float(self.cfg.beat.tempo_min)
        search_max_bpm = float(self.cfg.beat.tempo_max)

        # We search up to config tempo_max, but avoid returning very high
        # second guesses unless necessary.
        return_max_bpm = min(search_max_bpm, 210.0)

        raw = [
            (float(bpm), float(score))
            for bpm, score in zip(bpm_candidates, scores)
            if bpm > 0 and score > 0
        ]

        if not raw:
            return [120.0]

        def support_at(target: float) -> float:
            """
            Autocorrelation support around target tempo.
            Uses ±8%, matching the tempo evaluation tolerance.
            """
            if target <= 0:
                return 0.0

            support = 0.0

            for bpm, score in raw:
                rel_error = abs(bpm - target) / target

                if rel_error <= 0.08:
                    support += score * (1.0 - rel_error / 0.08)

            return support

        # Generate candidate base tempi from common metrical transformations.
        # These factors address observed validation errors such as:
        # 60 -> 90, 50 -> 75, 34 -> 102, 88 -> 29.
        factors = [
            1.0 / 3.0,
            1.0 / 2.0,
            2.0 / 3.0,
            3.0 / 4.0,
            1.0,
            4.0 / 3.0,
            3.0 / 2.0,
            2.0,
            3.0,
        ]

        base_candidates = []

        for bpm, _ in raw:
            for factor in factors:
                base = bpm * factor

                if base < min_bpm:
                    continue

                if 2.0 * base > return_max_bpm:
                    continue

                base_candidates.append(base)

        if not base_candidates:
            primary = float(bpm_candidates[0])

            if primary * 2.0 <= return_max_bpm:
                return sorted([round(primary, 3), round(primary * 2.0, 3)])

            if primary / 2.0 >= min_bpm:
                return sorted([round(primary / 2.0, 3), round(primary, 3)])

            return [round(primary, 3)]

        # Remove near-duplicate bases.
        base_candidates = sorted(base_candidates)
        unique_bases = []

        for base in base_candidates:
            if all(abs(base - old) / old > 0.03 for old in unique_bases):
                unique_bases.append(base)

        pair_scores = []

        for base in unique_bases:
            double = 2.0 * base

            s_base = support_at(base)
            s_double = support_at(double)

            # Related metrical supports.
            s_half = support_at(base / 2.0)
            s_three_half = support_at(base * 1.5)
            s_three = support_at(base * 3.0)
            s_two_thirds = support_at(base * 2.0 / 3.0)

            score = 0.0

            # Main pair evidence.
            score += 1.00 * s_base
            score += 1.00 * s_double

            # Bonus when both [base, 2*base] are supported.
            if s_base > 0 and s_double > 0:
                score += 0.75 * min(s_base, s_double)

            # Weaker evidence from related metrical levels.
            score += 0.35 * s_half
            score += 0.35 * s_three_half
            score += 0.25 * s_three
            score += 0.25 * s_two_thirds

            # Penalize one-sided pairs.
            if s_base == 0 or s_double == 0:
                score *= 0.75

            # Soft tempo prior.
            if 40.0 <= base <= 110.0:
                score *= 1.10
            elif 28.0 <= base < 40.0:
                score *= 0.95
            elif base > 110.0:
                score *= 0.90

            pair_scores.append(
                {
                    "score": score,
                    "base": base,
                    "s_base": s_base,
                    "s_double": s_double,
                }
            )

        pair_scores.sort(key=lambda x: x["score"], reverse=True)

        default = pair_scores[0]

        # Prefer plausible tactus pairs only when they have support for both
        # base and double. This prevents selecting [29, 59] when [89, 178]
        # is also clearly present.
        plausible_pairs = [
            item for item in pair_scores
            if (
                40.0 <= item["base"] <= 110.0
                and item["s_base"] > 0.20
                and item["s_double"] > 0.20
            )
        ]

        # Allow truly slow pairs only when both parts are strongly supported.
        slow_pairs = [
            item for item in pair_scores
            if (
                28.0 <= item["base"] < 40.0
                and item["s_base"] > 0.45
                and item["s_double"] > 0.45
            )
        ]

        best = default

        if plausible_pairs:
            candidate = plausible_pairs[0]

            if default["base"] < 40.0 and candidate["score"] >= 0.75 * default["score"]:
                best = candidate

            elif default["base"] > 110.0 and candidate["score"] >= 0.75 * default["score"]:
                best = candidate

            elif (
                40.0 <= default["base"] <= 110.0
                and candidate["score"] >= 1.25 * default["score"]
            ):
                best = candidate

        if slow_pairs:
            candidate = slow_pairs[0]

            if candidate["score"] >= 1.25 * best["score"]:
                best = candidate

        base = best["base"]
        tempos = [base, 2.0 * base]
        tempos = sorted([round(float(t), 3) for t in tempos])

        return tempos

    def estimate_primary(self, y: np.ndarray, sr: int) -> float:
        """
        Return the first submitted tempo estimate.

        Note:
        This is the lower tempo because estimate() returns sorted [T, 2T].
        For beat tracking, it is better to try both tempo guesses.
        """
        return self.estimate(y, sr)[0]

    def autocorrelation_tempo_candidates(
        self,
        result: DetectionFunctionResult,
        top_k: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute tempo candidates using manual autocorrelation.
        """
        d = np.asarray(result.values, dtype=np.float32)

        if len(d) < 3:
            return np.array([]), np.array([])

        d = d - np.mean(d)

        if np.allclose(d, 0):
            return np.array([]), np.array([])

        frame_rate = result.sr / result.hop_length

        min_bpm = float(self.cfg.beat.tempo_min)
        max_bpm = float(self.cfg.beat.tempo_max)

        min_lag = int(np.floor(frame_rate * 60.0 / max_bpm))
        max_lag = int(np.ceil(frame_rate * 60.0 / min_bpm))

        min_lag = max(min_lag, 1)
        max_lag = min(max_lag, len(d) - 1)

        if max_lag <= min_lag:
            return np.array([]), np.array([])

        lags = np.arange(min_lag, max_lag + 1)

        autocorr = np.zeros_like(lags, dtype=np.float32)

        for idx, lag in enumerate(lags):
            autocorr[idx] = np.sum(d[:-lag] * d[lag:])

        overlap = len(d) - lags
        autocorr = autocorr / np.maximum(overlap, 1)

        autocorr = np.maximum(autocorr, 0.0)

        if np.max(autocorr) > 0:
            autocorr = autocorr / np.max(autocorr)

        peak_indices = []

        for i in range(1, len(autocorr) - 1):
            if autocorr[i] >= autocorr[i - 1] and autocorr[i] >= autocorr[i + 1]:
                if autocorr[i] > 0:
                    peak_indices.append(i)

        if not peak_indices:
            peak_indices = [int(np.argmax(autocorr))]

        peak_indices = np.asarray(peak_indices, dtype=int)

        order = np.argsort(autocorr[peak_indices])[::-1]
        peak_indices = peak_indices[order][:top_k]

        best_lags = lags[peak_indices]
        bpms = 60.0 * frame_rate / best_lags
        scores = autocorr[peak_indices]

        return bpms, scores


class BeatTracker:
    """Track beats using manual tempo estimation and phase search."""

    def __init__(self, cfg=None):
        self.cfg = cfg or config
        self.fe = FeatureExtractor(self.cfg)
        self.tempo_estimator = TempoEstimator(self.cfg)

    def track(
        self,
        y: np.ndarray,
        sr: int,
        tempo: Optional[float] = None,
    ) -> Tuple[np.ndarray, float]:
        """
        Track beat times in seconds.

        Method:
        - use manual onset detection function
        - estimate tempo if not given
        - search beat-grid phase that best aligns with onset strength
        - optionally snap each beat to a nearby local maximum
        """
        result = self.fe.onset_strength(y, method=self.cfg.onset.method)

        if tempo is None:
            tempo = self.tempo_estimator.estimate_primary(y, sr)

        beat_frames = self._beat_frames_from_phase_search(result, tempo)
        beat_frames = self._snap_beats_to_local_peaks(result.values, beat_frames)

        beat_frames = beat_frames[
            (beat_frames >= 0) & (beat_frames < len(result.times))
        ]

        beat_times = result.times[beat_frames]

        return beat_times, float(tempo)

    def _beat_frames_from_phase_search(
        self,
        result: DetectionFunctionResult,
        tempo: float,
    ) -> np.ndarray:
        """
        Choose the beat-grid phase with the highest onset-function score.
        """
        d = np.asarray(result.values, dtype=np.float32)

        if len(d) == 0:
            return np.array([], dtype=int)

        frame_rate = result.sr / result.hop_length
        period_frames = int(round((60.0 / tempo) * frame_rate))
        period_frames = max(period_frames, 1)

        best_phase = 0
        best_score = -np.inf

        # Try every possible phase within one beat period.
        for phase in range(period_frames):
            frames = np.arange(phase, len(d), period_frames)
            if len(frames) == 0:
                continue

            score = np.sum(d[frames])

            # Mild preference for grids with enough beats.
            score = score / np.sqrt(len(frames))

            if score > best_score:
                best_score = score
                best_phase = phase

        beat_frames = np.arange(best_phase, len(d), period_frames)
        return beat_frames.astype(int)

    def _snap_beats_to_local_peaks(
        self,
        d: np.ndarray,
        beat_frames: np.ndarray,
        snap_window_sec: float = 0.07,
    ) -> np.ndarray:
        """
        Snap predicted beats to the strongest nearby onset-function frame.

        The default 70 ms window matches the beat evaluation tolerance.
        """
        if len(beat_frames) == 0:
            return beat_frames

        snap_radius = int(
            round(
                snap_window_sec
                * self.cfg.audio.sample_rate
                / self.cfg.audio.onset_hop_length
            )
        )
        snap_radius = max(snap_radius, 1)

        snapped = []

        for frame in beat_frames:
            left = max(0, frame - snap_radius)
            right = min(len(d), frame + snap_radius + 1)

            if right <= left:
                snapped.append(frame)
                continue

            local_best = left + int(np.argmax(d[left:right]))
            snapped.append(local_best)

        # Remove duplicates while preserving order.
        deduped = []
        for frame in snapped:
            if not deduped or frame != deduped[-1]:
                deduped.append(frame)

        return np.asarray(deduped, dtype=int)
    def track_dynamic_programming(
    self,
    y: np.ndarray,
    sr: int,
    tempo: float,
    transition_width: float = 0.20,
    transition_lambda: float = 0.8,
    activation_power: float = 1.5,
) -> Tuple[np.ndarray, float]:
        """
        Beat tracking via dynamic programming.

        This allows local tempo variation around the expected beat period.
        It is rule-safe: no ready-made beat tracker is used.
        """
        result = self.fe.onset_strength(y, method="beat_flux")
        d = np.asarray(result.values, dtype=np.float32)

        if len(d) == 0:
            return np.array([]), float(tempo)

        # Sharpen strong beat evidence slightly.
        d = np.maximum(d, 0.0)
        if np.max(d) > 0:
            d = d / np.max(d)
        d = d ** activation_power

        frame_rate = result.sr / result.hop_length
        expected_period = int(round((60.0 / tempo) * frame_rate))
        expected_period = max(expected_period, 1)

        min_period = int(round(expected_period * (1.0 - transition_width)))
        max_period = int(round(expected_period * (1.0 + transition_width)))

        min_period = max(min_period, 1)
        max_period = max(max_period, min_period + 1)

        n = len(d)

        dp = np.full(n, -np.inf, dtype=np.float32)
        backptr = np.full(n, -1, dtype=np.int32)

        # Allow starting anywhere, but prefer strong activations.
        dp[:] = d

        for t in range(n):
            best_score = d[t]
            best_prev = -1

            for period in range(min_period, max_period + 1):
                prev = t - period

                if prev < 0:
                    continue

                deviation = np.log(period / expected_period)
                transition_penalty = transition_lambda * (deviation ** 2)

                candidate_score = dp[prev] + d[t] - transition_penalty

                if candidate_score > best_score:
                    best_score = candidate_score
                    best_prev = prev

            dp[t] = best_score
            backptr[t] = best_prev

        # Pick best endpoint.
        end = int(np.argmax(dp))

        beat_frames = []
        cur = end

        while cur >= 0:
            beat_frames.append(cur)
            cur = int(backptr[cur])

        beat_frames = np.asarray(beat_frames[::-1], dtype=int)

        # Optional: remove too-early/too-late weak isolated frames.
        beat_frames = self._snap_beats_to_local_peaks(
            result.values,
            beat_frames,
            snap_window_sec=0.05,
        )

        beat_frames = beat_frames[
            (beat_frames >= 0) & (beat_frames < len(result.times))
        ]

        beat_times = result.times[beat_frames]

        return beat_times, float(tempo)


class Pipeline:
    """Complete rule-safe pipeline combining all manual detectors."""

    def __init__(self, cfg=None):
        self.cfg = cfg or config
        self.onset_detector = OnsetDetector(self.cfg)
        self.tempo_estimator = TempoEstimator(self.cfg)
        self.beat_tracker = BeatTracker(self.cfg)

    def process_file(
    self,
    y: np.ndarray,
    sr: int,
) -> Tuple[np.ndarray, np.ndarray, List[float]]:
        """
        Process one audio signal.

        Returns:
            onsets: onset times in seconds
            beats: beat times in seconds
            tempos: one or two tempo guesses
        """
        raw_tempos = self.tempo_estimator.estimate(y, sr)
        onsets = self.onset_detector.detect(y, sr)

        detection_result = self.beat_tracker.fe.onset_strength(
            y,
            method="beat_flux",
        )
        submitted_tempos = self.select_tempos_with_beat_feedback(
            y,
            sr,
            raw_tempos=raw_tempos,
            detection_result=detection_result,
            mode="max",
            alpha=0.65,
        )

        beat_tempo_candidates = []
        factors = (1.0, 1.5)

        for tempo in raw_tempos:
            for factor in factors:
                candidate = float(tempo) * factor

                if candidate < self.cfg.beat.tempo_min:
                    continue

                if candidate > self.cfg.beat.tempo_max:
                    continue

                if all(abs(candidate - old) / old > 0.04 for old in beat_tempo_candidates):
                    beat_tempo_candidates.append(candidate)

        beat_tempo_candidates = sorted(beat_tempo_candidates)

        best_beats = np.array([])
        best_score = -np.inf
        alpha = 0.65

        for tempo in beat_tempo_candidates:
            beats, _ = self.beat_tracker.track_dynamic_programming(
                y,
                sr,
                tempo=tempo,
                transition_width=0.10,
                transition_lambda=2.0,
                activation_power=1.0,
            )

            if len(beats) == 0:
                continue

            beat_frames = np.round(
                beats * detection_result.sr / detection_result.hop_length
            ).astype(int)

            beat_frames = beat_frames[
                (beat_frames >= 0)
                & (beat_frames < len(detection_result.values))
            ]

            if len(beat_frames) == 0:
                continue

            values = detection_result.values[beat_frames]
            score = np.sum(values) / (len(beat_frames) ** alpha)

            if score > best_score:
                best_score = score
                best_beats = beats

        if len(best_beats) == 0:
            best_beats, _ = self.beat_tracker.track_dynamic_programming(
                y,
                sr,
                tempo=raw_tempos[0],
                transition_width=0.10,
                transition_lambda=2.0,
                activation_power=1.0,
            )

        return onsets, best_beats, submitted_tempos

    def select_tempos_with_beat_feedback(
    self,
    y: np.ndarray,
    sr: int,
    raw_tempos: List[float],
    detection_result: DetectionFunctionResult,
    mode: str = "max",
    alpha: float = 0.65,
) -> List[float]:
        """
        Select submitted tempo pair using beat-DP evidence.

        This is used only for tempo output. Beat tracking can still use
        raw tempo candidates to avoid changing the beat pipeline.
        """
        factors = (1.0 / 3.0, 0.5, 0.75, 1.0, 1.5, 2.0)

        base = float(sorted(raw_tempos)[0])

        candidate_pairs = []

        for factor in factors:
            new_base = base * factor

            if new_base < self.cfg.beat.tempo_min:
                continue

            pair = [new_base]

            if 2.0 * new_base <= 210.0:
                pair.append(2.0 * new_base)

            pair = sorted([round(float(t), 3) for t in pair])

            if all(abs(pair[0] - old[0]) / old[0] > 0.04 for old in candidate_pairs):
                candidate_pairs.append(pair)

        if not candidate_pairs:
            return raw_tempos

        scored_pairs = []

        for pair in candidate_pairs:
            tempo_scores = []

            for tempo in pair:
                beats, _ = self.beat_tracker.track_dynamic_programming(
                    y,
                    sr,
                    tempo=float(tempo),
                    transition_width=0.10,
                    transition_lambda=2.0,
                    activation_power=1.0,
                )

                if len(beats) == 0:
                    tempo_scores.append(-np.inf)
                    continue

                beat_frames = np.round(
                    beats * detection_result.sr / detection_result.hop_length
                ).astype(int)

                beat_frames = beat_frames[
                    (beat_frames >= 0)
                    & (beat_frames < len(detection_result.values))
                ]

                if len(beat_frames) == 0:
                    tempo_scores.append(-np.inf)
                    continue

                values = detection_result.values[beat_frames]
                score = np.sum(values) / (len(beat_frames) ** alpha)

                tempo_scores.append(score)

            tempo_scores = np.asarray(tempo_scores, dtype=float)

            if mode == "max":
                pair_score = np.max(tempo_scores)
            elif mode == "balanced":
                if len(tempo_scores) == 1:
                    pair_score = tempo_scores[0]
                else:
                    pair_score = np.max(tempo_scores) + 0.35 * np.min(tempo_scores)
            elif mode == "upper":
                pair_score = tempo_scores[-1]
            else:
                raise ValueError(f"Unknown tempo feedback mode: {mode}")

            scored_pairs.append((pair, pair_score))

        best_pair, _ = max(scored_pairs, key=lambda x: x[1])

        return best_pair