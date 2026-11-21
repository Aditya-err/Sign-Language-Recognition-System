"""
Real-Time Gesture Inference Engine.
===================================

Author: Senior Python Developer & Computer Vision Engineer (Mentoring MCA Fresher)
Module: sign_language_v2.src.inference.predictor

Technical Overview for Interviews:
----------------------------------
This module provides low-latency inference on 126-dimensional normalized landmark vectors.

Key Design Patterns for Fast Inference:
  1. Direct Tensor Execution: Uses `self.model(tensor, training=False)` instead of
     `self.model.predict()`. As documented in our Phase 2.1 audit, `model.predict()` adds
     ~80-90 ms of unnecessary Keras dataset scaffolding per frame, whereas direct tensor
     calling runs in ~10 ms (or < 1 ms in pre-warmed C++ buffers).
  2. Safe Input Guarding: Validates vector length (126 floats), numerical stability
     (replaces NaN/Inf), and verifies hand presence (`is_hand_present`).
  3. No Prediction When Hands Missing: If input is an all-zero vector (meaning no hands
     were detected in the frame), the predictor returns a clean `has_hands=False` result
     without invoking the neural network.
"""

from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from src.core.normalization import DUAL_HAND_FEATURE_DIM, is_hand_present

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("predictor")


@dataclass
class PredictionResult:
    """
    Structured prediction container.

    Attributes:
        has_hands (bool): True if at least one valid hand landmark vector was detected.
        hand_type (str): 'dual', 'left_only', 'right_only', or 'none'.
        class_label (Optional[str]): Predicted sign class (e.g. 'A', '1'), or None if rejected.
        confidence (float): Softmax probability of top class [0.0, 1.0].
        top_k (List[Tuple[str, float]]): Top 3 candidate (class, confidence) pairs.
        raw_probabilities (Optional[np.ndarray]): Full softmax probability vector over 35 classes.
    """
    has_hands: bool
    hand_type: str
    class_label: Optional[str]
    confidence: float
    top_k: List[Tuple[str, float]]
    raw_probabilities: Optional[np.ndarray] = None

    @property
    def class_name(self) -> Optional[str]:
        """Compatibility alias for class_label."""
        return self.class_label


class SignPredictor:
    """
    Lightweight, high-performance inference engine for Indian Sign Language V2.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        label_encoder_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """
        Initializes the predictor, loads model weights and label mappings.

        Args:
            model_path: Path to best_model.keras (defaults to models/best_model.keras).
            label_encoder_path: Path to label_encoder.json (defaults to data/processed/label_encoder.json).
        """
        repo_root = Path(__file__).resolve().parents[2]

        if model_path is None:
            model_path = repo_root / "models" / "best_model.keras"
        else:
            model_path = Path(model_path)

        if label_encoder_path is None:
            label_encoder_path = repo_root / "data" / "processed" / "label_encoder.json"
        else:
            label_encoder_path = Path(label_encoder_path)

        if not model_path.is_file():
            raise FileNotFoundError(f"Trained model not found at: {model_path}")
        if not label_encoder_path.is_file():
            raise FileNotFoundError(f"Label encoder mapping not found at: {label_encoder_path}")

        logger.info("Loading Keras model from: %s", model_path)
        self.model = tf.keras.models.load_model(str(model_path))

        logger.info("Loading label encoder from: %s", label_encoder_path)
        with open(label_encoder_path, "r", encoding="utf-8") as f:
            encoder_data = json.load(f)

        self.classes: List[str] = encoder_data["classes"]
        self.num_classes: int = len(self.classes)
        self.idx_to_class = {int(k): v for k, v in encoder_data["idx_to_class"].items()}

        # Model validation check
        expected_out_dim = self.num_classes
        actual_out_dim = self.model.output_shape[-1]
        if actual_out_dim != expected_out_dim:
            raise ValueError(
                f"Model output dimension ({actual_out_dim}) does not match "
                f"label encoder classes count ({expected_out_dim})"
            )

        # Warm-up inference call to compile TensorFlow graph and initialize execution buffers
        dummy_input = np.zeros((1, DUAL_HAND_FEATURE_DIM), dtype=np.float32)
        _ = self.model(dummy_input, training=False)
        logger.info("Inference engine initialized successfully (%d classes)", self.num_classes)

    def predict(self, feature_vector: np.ndarray) -> PredictionResult:
        """
        Generates sign class prediction from a 126-D normalized feature vector.

        Args:
            feature_vector: Array of shape (126,) or (1, 126).

        Returns:
            PredictionResult dataclass with predicted sign and probabilities.
        """
        if feature_vector is None:
            return PredictionResult(
                has_hands=False,
                hand_type="none",
                class_label=None,
                confidence=0.0,
                top_k=[],
            )

        feat_flat = np.asarray(feature_vector, dtype=np.float32).flatten()

        if len(feat_flat) != DUAL_HAND_FEATURE_DIM:
            logger.warning("Invalid feature vector shape: %s", feat_flat.shape)
            return PredictionResult(
                has_hands=False,
                hand_type="none",
                class_label=None,
                confidence=0.0,
                top_k=[],
            )

        # Ensure finite values
        feat_flat = np.nan_to_num(feat_flat, nan=0.0, posinf=0.0, neginf=0.0)

        # Check hand presence in Left (0..62) and Right (63..125) slots
        left_present = is_hand_present(feat_flat[:63])
        right_present = is_hand_present(feat_flat[63:])

        if not left_present and not right_present:
            # No hands in frame: DO NOT run classifier, return immediately
            return PredictionResult(
                has_hands=False,
                hand_type="none",
                class_label=None,
                confidence=0.0,
                top_k=[],
            )

        if left_present and right_present:
            hand_type = "dual"
        elif left_present:
            hand_type = "left_only"
        else:
            hand_type = "right_only"

        # Direct tensor forward pass (avoids Keras predict overhead)
        x_tensor = tf.convert_to_tensor(feat_flat.reshape(1, DUAL_HAND_FEATURE_DIM), dtype=tf.float32)
        probs_tensor = self.model(x_tensor, training=False)
        probs = probs_tensor.numpy()[0]

        top_idx = int(np.argmax(probs))
        confidence = float(probs[top_idx])
        top_label = self.classes[top_idx]

        # Extract Top-3 candidates
        top_k_indices = np.argsort(probs)[-3:][::-1]
        top_k = [(self.classes[i], float(probs[i])) for i in top_k_indices]

        return PredictionResult(
            has_hands=True,
            hand_type=hand_type,
            class_label=top_label,
            confidence=confidence,
            top_k=top_k,
            raw_probabilities=probs,
        )
