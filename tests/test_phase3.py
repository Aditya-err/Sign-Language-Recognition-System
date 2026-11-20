"""
Unit and Integration Tests for Phase 3 Real-Time Recognition Modules.
=====================================================================

Author: Senior Python Developer & Computer Vision Engineer (Mentoring MCA Fresher)
Target: sign_language_v2 Phase 3 Modules (Predictor, Smoother, GestureState, TextBuilder)
Test Runner: pytest
"""

import numpy as np
import pytest

from src.core.normalization import DUAL_HAND_FEATURE_DIM
from src.inference.gesture_state import GestureState, GestureStateManager
from src.inference.predictor import PredictionResult, SignPredictor
from src.inference.smoothing import PredictionSmoother, SmoothedPrediction
from src.inference.text_builder import TextBuilder


@pytest.fixture(scope="module")
def shared_predictor():
    """Initializes the real trained model predictor once for test efficiency."""
    return SignPredictor()


# --- TASK 2 & 9: PREDICTOR TESTS ---

def test_predictor_model_dimensions_and_classes(shared_predictor):
    """Verifies predictor input dimension is 126 and output classes count is 35."""
    assert shared_predictor.num_classes == 35
    assert len(shared_predictor.classes) == 35
    assert "A" in shared_predictor.classes
    assert "1" in shared_predictor.classes


def test_predictor_missing_hand_safeguard(shared_predictor):
    """
    CRITICAL SAFETY TEST:
    All-zero vectors (meaning no hand detected) must NEVER generate a gesture prediction.
    """
    zero_vector = np.zeros(DUAL_HAND_FEATURE_DIM, dtype=np.float32)
    res = shared_predictor.predict(zero_vector)

    assert not res.has_hands
    assert res.hand_type == "none"
    assert res.class_label is None
    assert res.confidence == 0.0
    assert len(res.top_k) == 0


def test_predictor_valid_feature_prediction(shared_predictor):
    """Verifies that valid synthetic coordinates return structured PredictionResult."""
    # Synthetic vector with non-zero right hand coordinates (indices 63..125)
    sample = np.zeros(DUAL_HAND_FEATURE_DIM, dtype=np.float32)
    sample[63 + 9 * 3 + 1] = 1.0  # Middle MCP palm scale
    sample[63 + 1 * 3] = 0.5      # Thumb

    res = shared_predictor.predict(sample)

    assert res.has_hands
    assert res.hand_type == "right_only"
    assert res.class_label in shared_predictor.classes
    assert 0.0 <= res.confidence <= 1.0
    assert len(res.top_k) == 3
    assert res.top_k[0][0] == res.class_label


# --- TASK 4 & 9: PREDICTION SMOOTHING TESTS ---

def test_smoother_confidence_rejection():
    """Verifies that low-confidence predictions are filtered out by the smoother."""
    smoother = PredictionSmoother(window_size=5, min_confidence=0.70)

    # Low-confidence prediction (0.45 < 0.70)
    low_conf_pred = PredictionResult(
        has_hands=True,
        hand_type="right_only",
        class_label="A",
        confidence=0.45,
        top_k=[("A", 0.45)],
    )

    res = smoother.update(low_conf_pred)
    assert not res.is_stable
    assert res.class_label is None


def test_smoother_rolling_majority_vote():
    """Verifies majority voting and stability convergence across the temporal window."""
    smoother = PredictionSmoother(window_size=5, min_confidence=0.70, stability_threshold=0.60)

    high_conf_A = PredictionResult(
        has_hands=True,
        hand_type="right_only",
        class_label="A",
        confidence=0.90,
        top_k=[("A", 0.90)],
    )

    # Feed 4 consecutive 'A's
    for _ in range(4):
        res = smoother.update(high_conf_A)

    assert res.is_stable
    assert res.class_label == "A"
    assert res.confidence >= 0.85
    assert res.stability_score >= 0.80


def test_smoother_reset_on_hand_loss():
    """Verifies that buffer resets immediately when hands are dropped."""
    smoother = PredictionSmoother(window_size=5, min_confidence=0.70)

    # Establish stable state
    for _ in range(4):
        _ = smoother.update(PredictionResult(True, "right_only", "B", 0.95, [("B", 0.95)]))

    assert len(smoother.buffer) == 4

    # Hand lost
    no_hand = PredictionResult(False, "none", None, 0.0, [])
    res = smoother.update(no_hand)

    assert len(smoother.buffer) == 0
    assert not res.is_stable
    assert res.class_label is None


# --- TASK 5 & 9: GESTURE STATE MACHINE TESTS ---

def test_gesture_state_hold_and_commit():
    """Verifies FSM transitions: IDLE -> HOLDING -> COMMITTED -> LOCKED."""
    state_mgr = GestureStateManager(hold_frames_threshold=4, cooldown_frames=2)

    stable_A = SmoothedPrediction(
        class_label="A",
        confidence=0.92,
        is_stable=True,
        stability_score=1.0,
    )

    # Frame 1: Transitions to HOLDING
    out1 = state_mgr.update(stable_A)
    assert out1.state == GestureState.HOLDING
    assert out1.active_sign == "A"
    assert out1.emitted_char is None
    assert out1.hold_progress == 0.25

    # Frames 2 and 3: Progress increments
    _ = state_mgr.update(stable_A)
    _ = state_mgr.update(stable_A)

    # Frame 4: Threshold reached -> Emits character once!
    out4 = state_mgr.update(stable_A)
    assert out4.state == GestureState.COMMITTED
    assert out4.emitted_char == "A"
    assert out4.hold_progress == 1.0


def test_gesture_state_duplicate_prevention():
    """
    CRITICAL ANTI-DUPLICATE TEST:
    Continuously holding the same sign must NEVER emit duplicate characters.
    """
    state_mgr = GestureStateManager(hold_frames_threshold=3, cooldown_frames=2)
    stable_C = SmoothedPrediction("C", 0.95, True, 1.0)

    emitted_count = 0
    # Hold for 15 consecutive frames
    for _ in range(15):
        out = state_mgr.update(stable_C)
        if out.emitted_char == "C":
            emitted_count += 1

    assert emitted_count == 1, f"Expected exactly 1 emission, got {emitted_count}!"


def test_gesture_state_resets_on_lost_detection():
    """Verifies state returns to IDLE when hand detection is lost."""
    state_mgr = GestureStateManager(hold_frames_threshold=5)
    stable_D = SmoothedPrediction("D", 0.90, True, 1.0)

    # Hold partially
    _ = state_mgr.update(stable_D)
    _ = state_mgr.update(stable_D)
    assert state_mgr.state == GestureState.HOLDING

    # Lost detection
    lost = SmoothedPrediction(None, 0.0, False, 0.0)
    out = state_mgr.update(lost)

    assert out.state == GestureState.IDLE
    assert out.hold_progress == 0.0
    assert state_mgr.hold_count == 0


# --- TASK 7 & 9: TEXT BUILDER TESTS ---

def test_text_builder_editing_semantics():
    """Verifies characters, spaces, backspaces, clear, and bounds."""
    builder = TextBuilder(max_length=20)

    # Appending
    builder.add_character("I")
    builder.add_character("S")
    builder.add_character("L")
    assert builder.get_text() == "ISL"

    # Space insertion
    builder.add_space()
    builder.add_character("2")
    assert builder.get_text() == "ISL 2"

    # Double space prevention
    builder.add_space()
    builder.add_space()
    assert builder.get_text() == "ISL 2 "

    # Backspace
    builder.backspace()
    assert builder.get_text() == "ISL 2"
    builder.backspace()
    assert builder.get_text() == "ISL "

    # Clear
    builder.clear()
    assert builder.get_text() == ""
    assert builder.is_empty()
