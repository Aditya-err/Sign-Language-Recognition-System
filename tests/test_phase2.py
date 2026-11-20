"""
Unit and Integration Tests for Phase 2 Pipeline.
================================================

Author: Senior AI/ML Engineer (Mentoring MCA Fresher)
Target: sign_language_v2 Phase 2 Modules
Test Runner: pytest
"""

import numpy as np
import pytest
import tensorflow as tf

from src.core.normalization import (
    DUAL_HAND_FEATURE_DIM,
    LANDMARKS_PER_HAND,
    SINGLE_HAND_FEATURE_DIM,
    is_hand_present,
)
from src.data.augmentation import (
    apply_coordinate_jitter,
    apply_inplane_rotation,
    apply_scale_perturbation,
    augment_dual_hand_sample,
    augment_training_set,
)
from src.models.evaluate import compute_top_k_accuracy
from src.models.mlp import build_mlp_classifier


@pytest.fixture
def synthetic_present_hand():
    """Generates a realistic 63-D normalized hand feature vector."""
    landmarks = np.zeros((LANDMARKS_PER_HAND, 3), dtype=np.float32)
    # Wrist at (0, 0, 0)
    landmarks[0] = [0.0, 0.0, 0.0]
    # Middle MCP at (0.0, 1.0, 0.0) -> scaled palm
    landmarks[9] = [0.0, 1.0, 0.0]
    # Populate other joints
    for i in range(1, LANDMARKS_PER_HAND):
        if i != 9:
            landmarks[i] = [0.1 * i, 0.05 * i, 0.01 * i]
    return landmarks.flatten()


@pytest.fixture
def synthetic_absent_hand():
    """Returns a 63-D zero vector representing an absent hand."""
    return np.zeros(SINGLE_HAND_FEATURE_DIM, dtype=np.float32)


def test_augmentation_absent_hand_preservation(synthetic_absent_hand):
    """
    CRITICAL SAFETY TEST:
    An absent hand must remain strictly zero across all augmentation transforms.
    Injecting noise or rotating a missing hand would corrupt the model's semantics.
    """
    rng = np.random.RandomState(42)

    jittered = apply_coordinate_jitter(synthetic_absent_hand, sigma=0.05, random_state=rng)
    assert np.all(jittered == 0.0), "Jitter modified an absent hand!"

    scaled = apply_scale_perturbation(synthetic_absent_hand, scale_range=(0.8, 1.2), random_state=rng)
    assert np.all(scaled == 0.0), "Scale modified an absent hand!"

    rotated = apply_inplane_rotation(synthetic_absent_hand, max_angle_deg=15.0, random_state=rng)
    assert np.all(rotated == 0.0), "Rotation modified an absent hand!"


def test_augmentation_wrist_origin_preservation(synthetic_present_hand):
    """
    Verifies that Landmark 0 (wrist) remains anchored at origin (0, 0, 0)
    after jitter and rotation to preserve translation invariance.
    """
    rng = np.random.RandomState(42)

    jittered = apply_coordinate_jitter(synthetic_present_hand, sigma=0.02, random_state=rng)
    wrist_jittered = jittered[:3]
    assert np.allclose(wrist_jittered, [0.0, 0.0, 0.0], atol=1e-6), \
        f"Wrist origin moved after jitter: {wrist_jittered}"

    rotated = apply_inplane_rotation(synthetic_present_hand, max_angle_deg=10.0, random_state=rng)
    wrist_rotated = rotated[:3]
    assert np.allclose(wrist_rotated, [0.0, 0.0, 0.0], atol=1e-6), \
        f"Wrist origin moved after rotation: {wrist_rotated}"


def test_augment_dual_hand_sample(synthetic_present_hand, synthetic_absent_hand):
    """
    Verifies dual-hand augmentation preserves the absent hand while perturbing the present hand.
    """
    rng = np.random.RandomState(42)
    # Dual vector with Left Hand present, Right Hand absent (single-hand sign)
    dual_vec = np.concatenate([synthetic_present_hand, synthetic_absent_hand])
    assert dual_vec.shape == (DUAL_HAND_FEATURE_DIM,)

    augmented = augment_dual_hand_sample(dual_vec, random_state=rng)
    assert augmented.shape == (DUAL_HAND_FEATURE_DIM,)

    left_aug = augmented[:SINGLE_HAND_FEATURE_DIM]
    right_aug = augmented[SINGLE_HAND_FEATURE_DIM:]

    # Left hand should be perturbed but finite
    assert is_hand_present(left_aug)
    assert not np.array_equal(left_aug, synthetic_present_hand)
    assert np.all(np.isfinite(left_aug))

    # Right hand must remain completely zero
    assert not is_hand_present(right_aug)
    assert np.all(right_aug == 0.0)


def test_augment_training_set():
    """Verifies dataset-level augmentation size, label matching, and finite values."""
    N = 20
    X_dummy = np.random.normal(loc=0.0, scale=0.5, size=(N, DUAL_HAND_FEATURE_DIM)).astype(np.float32)
    y_dummy = np.random.randint(0, 5, size=(N,))

    X_aug, y_aug = augment_training_set(X_dummy, y_dummy, augmentation_factor=2, random_seed=42)

    assert len(X_aug) == N * (1 + 2) == 60
    assert len(y_aug) == 60
    assert X_aug.shape[1] == DUAL_HAND_FEATURE_DIM
    assert np.all(np.isfinite(X_aug))


def test_mlp_model_shapes_and_softmax():
    """
    Verifies that the MLP model compiles, accepts (batch, 126) input,
    outputs (batch, 35) valid probability distributions summing to 1.0.
    """
    batch_size = 8
    num_classes = 35
    model = build_mlp_classifier(input_dim=DUAL_HAND_FEATURE_DIM, num_classes=num_classes)

    dummy_input = np.random.uniform(-1.0, 1.0, size=(batch_size, DUAL_HAND_FEATURE_DIM)).astype(np.float32)
    preds = model(dummy_input, training=False).numpy()

    assert preds.shape == (batch_size, num_classes)
    # Check probabilities are in [0, 1]
    assert np.all(preds >= 0.0)
    assert np.all(preds <= 1.0 + 1e-6)
    # Softmax rows must sum to 1.0
    row_sums = np.sum(preds, axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-5)


def test_compute_top_k_accuracy():
    """Verifies the Top-K accuracy metric computation."""
    # 3 samples, 4 classes
    y_true = np.array([0, 2, 3])
    # Sample 0: class 0 is rank 1 (top-1 correct)
    # Sample 1: class 2 is rank 2 (top-1 wrong, top-3 correct)
    # Sample 2: class 3 is rank 4 (top-3 wrong)
    y_probs = np.array([
        [0.70, 0.10, 0.10, 0.10],  # rank order: 0, 1, 2, 3 -> Top 1: 0 (Correct)
        [0.40, 0.10, 0.35, 0.15],  # rank order: 0, 2, 3, 1 -> Top 1: 0, Top 3: 0, 2, 3 (Correct in Top 3)
        [0.40, 0.30, 0.20, 0.10],  # rank order: 0, 1, 2, 3 -> Top 3: 0, 1, 2 (Wrong, true is 3)
    ])

    top_1 = compute_top_k_accuracy(y_true, y_probs, k=1)
    top_3 = compute_top_k_accuracy(y_true, y_probs, k=3)

    assert pytest.approx(top_1, abs=1e-4) == 1.0 / 3.0
    assert pytest.approx(top_3, abs=1e-4) == 2.0 / 3.0
