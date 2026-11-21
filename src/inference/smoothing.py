"""
Prediction Stabilization and Temporal Smoothing Module.
======================================================

Author: Senior Python Developer & Computer Vision Engineer (Mentoring MCA Fresher)
Module: sign_language_v2.src.inference.smoothing

Technical Overview for Interviews:
----------------------------------
In real-time webcam sign recognition, single-frame predictions inevitably suffer from
high-frequency jitter and brief flickers caused by:
  1. Hand motion blur as fingers move into position.
  2. Temporary MediaPipe landmark jitter (1-2 pixels) between frames.
  3. Boundary ambiguities between kinematically similar signs (e.g. 'M' and 'N').

How Temporal Smoothing Works:
  - Rolling Window Buffer: Stores the last N predictions (e.g., N=7 frames, ~0.35 s at 20 FPS).
  - Confidence Threshold Filter: Predictions with confidence < threshold (default: 0.70, derived
    empirically from validation data) are rejected before polluting the buffer.
  - Majority Voting: Selects the most frequent class in the active window, weighted by confidence.
  - State Reset on Hand Loss: When no hand is detected, the history buffer is instantly cleared
    to prevent stale predictions from leaking across gestures.
"""

from collections import Counter, deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple

from src.inference.predictor import PredictionResult


@dataclass
class SmoothedPrediction:
    """
    Output container for smoothed prediction.

    Attributes:
        class_label (Optional[str]): Stabilized sign class, or None if uncertain/absent.
        confidence (float): Average confidence of the stabilized class [0.0, 1.0].
        is_stable (bool): True if the class dominates at least `stability_ratio` of the window.
        stability_score (float): Fraction of window agreeing on this class [0.0, 1.0].
    """
    class_label: Optional[str]
    confidence: float
    is_stable: bool
    stability_score: float

    @property
    def class_name(self) -> Optional[str]:
        """Compatibility alias for class_label."""
        return self.class_label


class PredictionSmoother:
    """
    Temporal buffer that stabilizes frame-by-frame predictions using confidence-aware voting.
    """

    def __init__(
        self,
        window_size: int = 7,
        min_confidence: float = 0.70,
        stability_threshold: float = 0.60,
    ) -> None:
        """
        Initializes the prediction smoother.

        Args:
            window_size: Number of consecutive frames to track (default: 7).
            min_confidence: Minimum softmax probability to accept a frame (default: 0.70).
            stability_threshold: Required fraction of window consensus to consider stable (default: 0.60).
        """
        self.window_size = window_size
        self.min_confidence = min_confidence
        self.stability_threshold = stability_threshold

        # Queue storing tuples of (class_label, confidence)
        self.buffer: Deque[Tuple[str, float]] = deque(maxlen=window_size)

    def update(self, pred: PredictionResult) -> SmoothedPrediction:
        """
        Processes a raw frame prediction and returns the temporally smoothed result.

        Args:
            pred: PredictionResult from SignPredictor.

        Returns:
            SmoothedPrediction dataclass.
        """
        # Rule 1: Reset history instantly if hand is lost
        if not pred.has_hands or pred.class_label is None:
            self.reset()
            return SmoothedPrediction(
                class_label=None,
                confidence=0.0,
                is_stable=False,
                stability_score=0.0,
            )

        # Rule 2: Reject low-confidence / uncertain predictions
        if pred.confidence < self.min_confidence:
            # If buffer is empty, return None; otherwise evaluate existing buffer
            if len(self.buffer) == 0:
                return SmoothedPrediction(
                    class_label=None,
                    confidence=pred.confidence,
                    is_stable=False,
                    stability_score=0.0,
                )
        else:
            # Append high-confidence prediction to rolling window
            self.buffer.append((pred.class_label, pred.confidence))

        if len(self.buffer) == 0:
            return SmoothedPrediction(
                class_label=None,
                confidence=0.0,
                is_stable=False,
                stability_score=0.0,
            )

        # Rule 3: Weighted voting across active window
        class_weights = Counter()
        class_confs = Counter()
        class_counts = Counter()

        for label, conf in self.buffer:
            class_weights[label] += conf
            class_confs[label] += conf
            class_counts[label] += 1

        top_class, _ = class_weights.most_common(1)[0]
        count = class_counts[top_class]
        avg_conf = class_confs[top_class] / max(count, 1)
        stability_score = count / len(self.buffer)
        is_stable = (stability_score >= self.stability_threshold) and (len(self.buffer) >= 3)

        return SmoothedPrediction(
            class_label=top_class if is_stable else None,
            confidence=avg_conf,
            is_stable=is_stable,
            stability_score=stability_score,
        )

    def reset(self) -> None:
        """Clears the temporal window buffer."""
        self.buffer.clear()
