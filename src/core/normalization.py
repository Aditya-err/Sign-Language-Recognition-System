"""
Landmark Normalization Module for Indian Sign Language (ISL) Recognition.
========================================================================

Author: Senior AI/ML Engineer (Mentoring MCA Fresher)
Module: sign_language_v2.src.core.normalization

Technical Overview for Interviews:
----------------------------------
Raw MediaPipe hand landmarks are given in screen-normalized coordinates (x, y in [0, 1]).
This raw format has two critical flaws for machine learning:
  1. Position Sensitivity: A sign performed on the left of the webcam has different
     coordinates than the same sign performed on the right.
  2. Scale Sensitivity: A sign performed close to the webcam has larger coordinates
     than the same sign performed further away.

This module applies deterministic mathematical transformations to achieve:
  - Translation Invariance: The wrist (Landmark 0) is moved to origin (0, 0, 0).
  - Scale Invariance: All coordinates are divided by the rigid palm length
    (Euclidean distance between Wrist [0] and Middle Finger MCP Joint [9]).

Because Indian Sign Language (ISL) uses both single-hand signs (e.g., 'C', '1')
and two-hand signs (e.g., 'A', 'B'), this module provides:
  - normalize_single_hand: Produces a 63-dimensional feature vector (21 landmarks x 3).
  - normalize_dual_hands: Produces a 126-dimensional feature vector (63 Left + 63 Right).
"""

from typing import List, Optional, Tuple, Union
import numpy as np

# Constant dimensions
LANDMARKS_PER_HAND = 21
COORDINATES_PER_LANDMARK = 3
SINGLE_HAND_FEATURE_DIM = LANDMARKS_PER_HAND * COORDINATES_PER_LANDMARK  # 63
DUAL_HAND_FEATURE_DIM = SINGLE_HAND_FEATURE_DIM * 2                      # 126

# Key landmark indices based on MediaPipe Hand Topology
WRIST_IDX = 0            # Origin reference point
MIDDLE_MCP_IDX = 9       # Rigid palm reference point (Middle finger base knuckle)
EPSILON = 1e-6           # Prevents division by zero for zero-scale edge cases


def normalize_single_hand(
    landmarks: Optional[Union[List[Tuple[float, float, float]], np.ndarray]]
) -> np.ndarray:
    """
    Normalizes a single hand's 21 landmarks into a 63-dimensional feature vector.
    
    Transformation Steps:
      1. Validation: Checks for None, empty lists, or invalid shapes.
      2. Translation: Subtracts wrist coordinate (p0) from all 21 landmarks.
         Wrist becomes exactly (0.0, 0.0, 0.0).
      3. Scale: Computes Euclidean distance s = ||p9 - p0||_2.
         Divides all coordinates by s (if s >= 1e-6).
      4. Safety: Replaces any potential NaN/inf with 0.0.
    
    Args:
        landmarks: List or array of 21 (x, y, z) landmark coordinates.
        
    Returns:
        np.ndarray: 1D array of shape (63,) with dtype float32.
                    Returns zeros(63) if input is None, empty, or invalid.
    """
    if landmarks is None:
        return np.zeros(SINGLE_HAND_FEATURE_DIM, dtype=np.float32)

    # Convert input to numpy array with float64 precision for numerical stability
    try:
        coords = np.asarray(landmarks, dtype=np.float64)
    except (ValueError, TypeError):
        return np.zeros(SINGLE_HAND_FEATURE_DIM, dtype=np.float32)

    # Validate shape: must be either (21, 3) or (63,)
    if coords.size != SINGLE_HAND_FEATURE_DIM:
        return np.zeros(SINGLE_HAND_FEATURE_DIM, dtype=np.float32)

    coords = coords.reshape((LANDMARKS_PER_HAND, COORDINATES_PER_LANDMARK))

    # Check for NaN or Inf in raw input
    if not np.all(np.isfinite(coords)):
        return np.zeros(SINGLE_HAND_FEATURE_DIM, dtype=np.float32)

    # Step 1: Wrist-Centered Translation
    # Center all points so the wrist (Landmark 0) sits at (0, 0, 0)
    wrist = coords[WRIST_IDX].copy()
    translated = coords - wrist

    # Step 2: Palm-Scale Invariance
    # Reference distance: distance from wrist (now 0,0,0) to middle finger base MCP (Landmark 9)
    # The middle MCP is anatomically rigid relative to the wrist (palm size does not bend).
    middle_mcp = translated[MIDDLE_MCP_IDX]
    palm_scale = float(np.linalg.norm(middle_mcp))

    # Step 3: Normalization with Zero-Scale Guard
    if palm_scale < EPSILON:
        # If all points are on top of each other (collapsed/dummy points), return translated
        normalized = translated
    else:
        normalized = translated / palm_scale

    # Flatten into a 1D vector of length 63 and cast to float32
    feature_vector = normalized.flatten().astype(np.float32)

    # Final sanity check: ensure strictly finite values
    return np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)


def normalize_dual_hands(
    left_landmarks: Optional[Union[List[Tuple[float, float, float]], np.ndarray]] = None,
    right_landmarks: Optional[Union[List[Tuple[float, float, float]], np.ndarray]] = None,
) -> np.ndarray:
    """
    Normalizes dual-hand landmarks into a unified 126-dimensional feature vector.
    
    Feature Layout:
      - Indices   0 to  62: Normalized Left Hand (63 floats)
      - Indices  63 to 125: Normalized Right Hand (63 floats)
      
    If only one hand is present (e.g., for single-hand signs like 'C' or '1'):
      - The present hand is normalized to its own wrist and palm scale.
      - The absent hand is cleanly populated with zeros.
      
    This enables the neural network to learn both single-handed and two-handed
    Indian Sign Language gestures in a single consistent feature representation.
    
    Args:
        left_landmarks: Landmarks for Left hand, or None if absent.
        right_landmarks: Landmarks for Right hand, or None if absent.
        
    Returns:
        np.ndarray: 1D array of shape (126,) with dtype float32.
    """
    left_features = normalize_single_hand(left_landmarks)
    right_features = normalize_single_hand(right_landmarks)

    dual_vector = np.concatenate([left_features, right_features], axis=0)
    return dual_vector.astype(np.float32)


def is_hand_present(hand_vector: np.ndarray) -> bool:
    """
    Helper to check if a 63-dimensional normalized hand vector contains valid data
    (i.e., not all zeros).
    """
    if hand_vector is None or len(hand_vector) != SINGLE_HAND_FEATURE_DIM:
        return False
    return bool(np.any(hand_vector != 0.0))
