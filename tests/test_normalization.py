"""
Unit Tests for Landmark Normalization Module.
=============================================

Author: Senior AI/ML Engineer (Mentoring MCA Fresher)
Target: sign_language_v2.src.core.normalization
Test Runner: pytest
"""

import numpy as np
import pytest

from src.core.normalization import (
    normalize_single_hand,
    normalize_dual_hands,
    is_hand_present,
    SINGLE_HAND_FEATURE_DIM,
    DUAL_HAND_FEATURE_DIM,
)


@pytest.fixture
def sample_hand_landmarks():
    """
    Creates a realistic synthetic 21-landmark hand configuration.
    Wrist (0) at (0.5, 0.8, 0.0).
    Middle finger MCP (9) at (0.5, 0.5, 0.0) -> palm distance = 0.3.
    Other landmarks distributed realistically.
    """
    landmarks = np.zeros((21, 3), dtype=np.float32)
    landmarks[0] = [0.5, 0.8, 0.0]   # Wrist
    landmarks[9] = [0.5, 0.5, 0.0]   # Middle MCP (distance = 0.3)
    
    # Assign distinct values to other landmarks
    for i in range(1, 21):
        if i != 9:
            landmarks[i] = [0.5 + (i * 0.01), 0.8 - (i * 0.02), i * 0.005]
            
    return landmarks


def test_output_dimensions(sample_hand_landmarks):
    """Verifies that normalized outputs match exact expected feature dimensions."""
    single_out = normalize_single_hand(sample_hand_landmarks)
    assert single_out.shape == (SINGLE_HAND_FEATURE_DIM,), \
        f"Expected single hand shape ({SINGLE_HAND_FEATURE_DIM},), got {single_out.shape}"
    assert single_out.dtype == np.float32

    dual_out = normalize_dual_hands(sample_hand_landmarks, sample_hand_landmarks)
    assert dual_out.shape == (DUAL_HAND_FEATURE_DIM,), \
        f"Expected dual hand shape ({DUAL_HAND_FEATURE_DIM},), got {dual_out.shape}"
    assert dual_out.dtype == np.float32


def test_wrist_centering(sample_hand_landmarks):
    """Verifies that Landmark 0 (the wrist) is always translated to (0.0, 0.0, 0.0)."""
    normalized = normalize_single_hand(sample_hand_landmarks)
    wrist_coords = normalized[0:3]
    np.testing.assert_allclose(
        wrist_coords, [0.0, 0.0, 0.0], atol=1e-6,
        err_msg="Wrist coordinates must be identically centered at (0.0, 0.0, 0.0)"
    )


def test_translation_invariance(sample_hand_landmarks):
    """
    Moving the hand across the screen (adding delta X, Y, Z) must NOT
    change the normalized feature vector within numerical float32 precision.
    """
    base_normalized = normalize_single_hand(sample_hand_landmarks)

    # 1. Realistic cross-screen translation in camera frame (e.g. shift by +0.35, -0.25, +0.10)
    screen_shift = sample_hand_landmarks + np.array([0.35, -0.25, 0.10], dtype=np.float32)
    screen_normalized = normalize_single_hand(screen_shift)
    np.testing.assert_allclose(
        screen_normalized, base_normalized, atol=1e-5,
        err_msg="Normalized features must be invariant to frame-space translation."
    )

    # 2. Large translation stress-test (e.g. +10.0, -15.0) within float32 precision tolerance (1e-4)
    large_shift = sample_hand_landmarks + np.array([10.0, -15.0, 5.0], dtype=np.float32)
    large_normalized = normalize_single_hand(large_shift)
    np.testing.assert_allclose(
        large_normalized, base_normalized, atol=1e-4,
        err_msg="Normalized features must be invariant to large spatial translation."
    )


def test_scale_invariance(sample_hand_landmarks):
    """
    Standing closer or further from the camera (scaling all coordinates by 2.5x or 0.4x)
    must NOT change the normalized feature vector.
    """
    base_normalized = normalize_single_hand(sample_hand_landmarks)
    
    # Scale up by 2.5x
    scaled_up = sample_hand_landmarks * 2.5
    scaled_up_normalized = normalize_single_hand(scaled_up)
    np.testing.assert_allclose(
        scaled_up_normalized, base_normalized, atol=1e-5,
        err_msg="Normalized features must be invariant to scale expansion (2.5x)."
    )
    
    # Scale down by 0.4x
    scaled_down = sample_hand_landmarks * 0.4
    scaled_down_normalized = normalize_single_hand(scaled_down)
    np.testing.assert_allclose(
        scaled_down_normalized, base_normalized, atol=1e-5,
        err_msg="Normalized features must be invariant to scale contraction (0.4x)."
    )


def test_invalid_and_none_inputs():
    """Ensures safe handling of None, empty lists, and invalid input structures."""
    # None input
    res_none = normalize_single_hand(None)
    assert res_none.shape == (63,)
    assert np.all(res_none == 0.0)
    
    # Empty list
    res_empty = normalize_single_hand([])
    assert res_empty.shape == (63,)
    assert np.all(res_empty == 0.0)
    
    # Malformed shape (only 5 points instead of 21)
    res_malformed = normalize_single_hand(np.zeros((5, 3)))
    assert res_malformed.shape == (63,)
    assert np.all(res_malformed == 0.0)
    
    # Non-numeric garbage
    res_garbage = normalize_single_hand("invalid_input")  # type: ignore
    assert res_garbage.shape == (63,)
    assert np.all(res_garbage == 0.0)


def test_zero_scale_handling():
    """
    When all 21 points are identical (scale = 0), division by zero must be prevented
    without throwing an exception or producing NaN/Inf.
    """
    all_zero_points = np.zeros((21, 3), dtype=np.float32)
    res_zero = normalize_single_hand(all_zero_points)
    assert res_zero.shape == (63,)
    assert np.all(np.isfinite(res_zero))
    assert np.all(res_zero == 0.0)
    
    # All points coincident at (10.0, 10.0, 10.0)
    all_coincident = np.full((21, 3), 10.0, dtype=np.float32)
    res_coincident = normalize_single_hand(all_coincident)
    assert res_coincident.shape == (63,)
    assert np.all(np.isfinite(res_coincident))
    assert np.all(res_coincident == 0.0)


def test_nan_and_infinite_rejection():
    """If NaN or Inf coordinates are in the raw data, the module must safely return zeros."""
    corrupted = np.ones((21, 3), dtype=np.float32)
    corrupted[5, 1] = np.nan
    corrupted[10, 2] = np.inf
    
    res = normalize_single_hand(corrupted)
    assert res.shape == (63,)
    assert np.all(np.isfinite(res))
    assert np.all(res == 0.0)


def test_dual_hand_layout(sample_hand_landmarks):
    """
    Verifies that dual hand normalization places Left hand in [0:63]
    and Right hand in [63:126], and correctly handles asymmetric single-hand inputs.
    """
    # Case 1: Both hands present
    left_hand = sample_hand_landmarks
    right_hand = sample_hand_landmarks * 1.5  # Slightly different scale
    dual = normalize_dual_hands(left_hand, right_hand)
    
    assert dual.shape == (126,)
    assert is_hand_present(dual[0:63]) is True
    assert is_hand_present(dual[63:126]) is True
    
    # Case 2: Only Right hand present (Left is None)
    right_only = normalize_dual_hands(left_landmarks=None, right_landmarks=right_hand)
    assert np.all(right_only[0:63] == 0.0)
    assert is_hand_present(right_only[0:63]) is False
    assert is_hand_present(right_only[63:126]) is True
    
    # Case 3: Both hands None
    both_none = normalize_dual_hands(None, None)
    assert np.all(both_none == 0.0)
    assert is_hand_present(both_none[0:63]) is False
    assert is_hand_present(both_none[63:126]) is False


def test_deterministic_repeatability(sample_hand_landmarks):
    """Calling the normalization function multiple times must return identical arrays."""
    run1 = normalize_single_hand(sample_hand_landmarks)
    run2 = normalize_single_hand(sample_hand_landmarks)
    np.testing.assert_array_equal(run1, run2)
