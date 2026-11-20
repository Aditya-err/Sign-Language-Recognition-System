"""
Landmark Feature Augmentation Module.
====================================

Author: Senior AI/ML Engineer (Mentoring MCA Fresher)
Module: sign_language_v2.src.data.augmentation

Technical Overview for Interviews:
----------------------------------
In standard computer vision, data augmentation operates on pixel grids (random crops,
color jitter, blurring). However, because our representation is a 126-dimensional
geometric landmark coordinate vector, image-space augmentations are inapplicable.

Instead, this module operates in Coordinate Space:
  1. Gaussian Coordinate Jitter: Adds subtle noise (sigma=0.01) to joint coordinates,
     simulating minor hand tremors and camera sensor noise.
  2. Scale Perturbation: Simulates minor distance variations relative to the webcam.
  3. In-Plane Rotation: Rotates 2D joint coordinates in the camera plane by +/- 8 degrees,
     simulating natural hand tilt during signing.

CRITICAL ARCHITECTURAL RULES:
  1. Preserves Absent Hands: If a hand is absent (zero-padded 63-D vector), it MUST
     remain exactly zero! Adding noise to an absent hand creates synthetic ghosts.
  2. Preserves Normalized Invariants: Landmark 0 (wrist) is the origin reference point (0,0,0)
     and must remain anchored at the origin.
  3. Strictly Training Only: Augmentation is NEVER applied to validation or test data!
"""

import math
from typing import Optional, Tuple
import numpy as np

from src.core.normalization import (
    DUAL_HAND_FEATURE_DIM,
    LANDMARKS_PER_HAND,
    SINGLE_HAND_FEATURE_DIM,
    is_hand_present,
)


def apply_coordinate_jitter(
    hand_vector: np.ndarray,
    sigma: float = 0.01,
    random_state: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """
    Adds zero-mean Gaussian jitter to non-wrist joint coordinates of a single hand.

    Args:
        hand_vector: 63-D normalized hand vector.
        sigma: Standard deviation of Gaussian noise.
        random_state: Optional numpy RandomState for reproducibility.

    Returns:
        Augmented 63-D hand vector (or original zeros if hand was absent).
    """
    if not is_hand_present(hand_vector):
        return np.zeros_like(hand_vector)

    rng = random_state if random_state is not None else np.random

    # Reshape to (21, 3)
    coords = hand_vector.reshape((LANDMARKS_PER_HAND, 3)).copy()

    # Landmark 0 (Wrist) is anchored at (0, 0, 0); only jitter finger joints 1..20
    noise = rng.normal(loc=0.0, scale=sigma, size=(LANDMARKS_PER_HAND - 1, 3)).astype(np.float32)
    coords[1:] += noise

    return coords.flatten()


def apply_scale_perturbation(
    hand_vector: np.ndarray,
    scale_range: Tuple[float, float] = (0.95, 1.05),
    random_state: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """
    Scales hand landmark coordinates by a factor drawn uniformly from scale_range.

    Args:
        hand_vector: 63-D normalized hand vector.
        scale_range: Min and max scaling factor.
        random_state: Optional numpy RandomState.

    Returns:
        Augmented 63-D hand vector.
    """
    if not is_hand_present(hand_vector):
        return np.zeros_like(hand_vector)

    rng = random_state if random_state is not None else np.random
    scale_factor = float(rng.uniform(scale_range[0], scale_range[1]))

    return (hand_vector * scale_factor).astype(np.float32)


def apply_inplane_rotation(
    hand_vector: np.ndarray,
    max_angle_deg: float = 8.0,
    random_state: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """
    Rotates 2D (x, y) joint coordinates around the wrist origin in the camera plane.

    Args:
        hand_vector: 63-D normalized hand vector.
        max_angle_deg: Maximum rotation angle in degrees (+/-).
        random_state: Optional numpy RandomState.

    Returns:
        Augmented 63-D hand vector.
    """
    if not is_hand_present(hand_vector):
        return np.zeros_like(hand_vector)

    rng = random_state if random_state is not None else np.random
    theta_deg = float(rng.uniform(-max_angle_deg, max_angle_deg))
    theta_rad = math.radians(theta_deg)

    cos_t = math.cos(theta_rad)
    sin_t = math.sin(theta_rad)

    coords = hand_vector.reshape((LANDMARKS_PER_HAND, 3)).copy()
    x = coords[:, 0].copy()
    y = coords[:, 1].copy()

    # Apply standard 2D rotation matrix
    coords[:, 0] = x * cos_t - y * sin_t
    coords[:, 1] = x * sin_t + y * cos_t

    return coords.flatten().astype(np.float32)


def augment_dual_hand_sample(
    sample_126: np.ndarray,
    jitter_sigma: float = 0.01,
    scale_range: Tuple[float, float] = (0.95, 1.05),
    max_rotation_deg: float = 8.0,
    random_state: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """
    Applies coordinate-space augmentation to a 126-D dual-hand feature vector.

    Args:
        sample_126: Original 126-D normalized vector (63 Left + 63 Right).
        jitter_sigma: Gaussian noise standard deviation.
        scale_range: Scale perturbation range.
        max_rotation_deg: Maximum in-plane rotation angle.
        random_state: Optional numpy RandomState.

    Returns:
        Augmented 126-D vector.
    """
    if len(sample_126) != DUAL_HAND_FEATURE_DIM:
        raise ValueError(f"Expected shape ({DUAL_HAND_FEATURE_DIM},), got {sample_126.shape}")

    left_part = sample_126[:SINGLE_HAND_FEATURE_DIM]
    right_part = sample_126[SINGLE_HAND_FEATURE_DIM:]

    # Augment Left hand if present
    if is_hand_present(left_part):
        left_aug = apply_coordinate_jitter(left_part, sigma=jitter_sigma, random_state=random_state)
        left_aug = apply_scale_perturbation(left_aug, scale_range=scale_range, random_state=random_state)
        left_aug = apply_inplane_rotation(left_aug, max_angle_deg=max_rotation_deg, random_state=random_state)
    else:
        left_aug = np.zeros(SINGLE_HAND_FEATURE_DIM, dtype=np.float32)

    # Augment Right hand if present
    if is_hand_present(right_part):
        right_aug = apply_coordinate_jitter(right_part, sigma=jitter_sigma, random_state=random_state)
        right_aug = apply_scale_perturbation(right_aug, scale_range=scale_range, random_state=random_state)
        right_aug = apply_inplane_rotation(right_aug, max_angle_deg=max_rotation_deg, random_state=random_state)
    else:
        right_aug = np.zeros(SINGLE_HAND_FEATURE_DIM, dtype=np.float32)

    return np.concatenate([left_aug, right_aug]).astype(np.float32)


def augment_training_set(
    X_train: np.ndarray,
    y_train: np.ndarray,
    augmentation_factor: int = 1,
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Augments the training partition by generating synthetic coordinate variations.

    Args:
        X_train: Training features array of shape (N, 126).
        y_train: Training labels array of shape (N,).
        augmentation_factor: Number of synthetic copies per original training sample.
        random_seed: Random seed for reproducibility.

    Returns:
        Tuple of (X_augmented, y_augmented) with shape (N * (1 + factor), 126).
    """
    if augmentation_factor <= 0:
        return X_train.copy(), y_train.copy()

    rng = np.random.RandomState(random_seed)
    augmented_X_list = [X_train]
    augmented_y_list = [y_train]

    for _ in range(augmentation_factor):
        aug_batch = np.zeros_like(X_train)
        for i in range(len(X_train)):
            aug_batch[i] = augment_dual_hand_sample(X_train[i], random_state=rng)
        augmented_X_list.append(aug_batch)
        augmented_y_list.append(y_train)

    X_augmented = np.concatenate(augmented_X_list, axis=0)
    y_augmented = np.concatenate(augmented_y_list, axis=0)

    # Shuffle augmented set together
    shuffle_indices = rng.permutation(len(X_augmented))
    return X_augmented[shuffle_indices], y_augmented[shuffle_indices]
