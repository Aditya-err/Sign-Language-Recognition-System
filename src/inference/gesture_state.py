"""
Gesture State Machine and Commit Controller.
============================================

Author: Senior Python Developer & Computer Vision Engineer (Mentoring MCA Fresher)
Module: sign_language_v2.src.inference.gesture_state

Technical Overview for Interviews:
----------------------------------
In real-time fingerspelling, users hold a sign in front of the camera while intending
to type a single character. Without a state machine:
  - If a model runs at 20 FPS, holding sign 'A' for 2 seconds would produce 40 'A's! ("AAAA...")
  - When the hand transitions from 'A' to 'B', intermediate noisy frames would insert garbage characters.

The 4-State Gesture Controller:
  1. IDLE: No hand detected or hand is moving into position (hold_count = 0).
  2. HOLDING: User is steadily holding a specific sign class. Progress bar increments (0% -> 100%).
  3. COMMITTED: User has held the sign steadily for `hold_frames_threshold` consecutive frames.
     The character is emitted EXACTLY ONCE.
  4. LOCKED / COOLDOWN: Prevents any further character emissions as long as the user continues
     holding the same sign. A new sign can only be emitted after the user releases the hand
     OR transitions to a different sign for at least `cooldown_frames`.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from src.inference.smoothing import SmoothedPrediction


class GestureState(Enum):
    """Finite State Machine states for sign commitment."""
    IDLE = "IDLE"              # Waiting for valid stable sign
    HOLDING = "HOLDING"        # User is steadily forming sign (progressing)
    COMMITTED = "COMMITTED"    # Threshold reached; character emitted once
    LOCKED = "LOCKED"          # Sign emitted; locked until release or transition


@dataclass
class GestureStateOutput:
    """
    Status output from the gesture state machine.

    Attributes:
        state (GestureState): Current FSM state.
        active_sign (Optional[str]): Current sign candidate.
        hold_progress (float): Hold progress [0.0, 1.0] for HUD progress bar.
        emitted_char (Optional[str]): Single character emitted on transition to COMMITTED, or None.
        status_message (str): Human-readable status description for UI display.
    """
    state: GestureState
    active_sign: Optional[str]
    hold_progress: float
    emitted_char: Optional[str]
    status_message: str

    @property
    def current_state(self) -> GestureState:
        """Compatibility alias for self.state."""
        return self.state

    @property
    def committed_character(self) -> Optional[str]:
        """Compatibility alias for self.emitted_char."""
        return self.emitted_char


class GestureStateManager:
    """
    Lightweight FSM controlling steady hold detection and single-character emission.
    """

    def __init__(
        self,
        hold_frames_threshold: int = 12,
        cooldown_frames: int = 6,
    ) -> None:
        """
        Initializes the gesture state manager.

        Args:
            hold_frames_threshold: Number of consecutive stable frames required to commit (default: 12, ~0.6 s).
            cooldown_frames: Minimum frames required before a new sign can begin holding (default: 6).
        """
        self.hold_frames_threshold = hold_frames_threshold
        self.cooldown_frames = cooldown_frames

        self.state = GestureState.IDLE
        self.current_candidate: Optional[str] = None
        self.hold_count: int = 0
        self.cooldown_count: int = 0
        self.last_emitted_sign: Optional[str] = None

    def update(self, smoothed: SmoothedPrediction) -> GestureStateOutput:
        """
        Updates the state machine with the latest smoothed prediction.

        Args:
            smoothed: SmoothedPrediction from PredictionSmoother.

        Returns:
            GestureStateOutput with progress and optional emitted character.
        """
        # Case 1: Hand lost or prediction unstable
        if not smoothed.is_stable or smoothed.class_label is None:
            self.reset_hold()
            # If we were locked and hand dropped, return to IDLE
            if self.state == GestureState.LOCKED:
                self.state = GestureState.IDLE
                self.last_emitted_sign = None

            return GestureStateOutput(
                state=GestureState.IDLE,
                active_sign=None,
                hold_progress=0.0,
                emitted_char=None,
                status_message="Show gesture to camera...",
            )

        sign = smoothed.class_label

        # Case 2: In Cooldown
        if self.cooldown_count > 0:
            self.cooldown_count -= 1
            return GestureStateOutput(
                state=GestureState.LOCKED,
                active_sign=sign,
                hold_progress=0.0,
                emitted_char=None,
                status_message="Cooldown...",
            )

        # Case 3: Same sign already committed and currently locked
        if self.state == GestureState.LOCKED:
            if sign == self.last_emitted_sign:
                return GestureStateOutput(
                    state=GestureState.LOCKED,
                    active_sign=sign,
                    hold_progress=1.0,
                    emitted_char=None,
                    status_message=f"Hold '{sign}' (Release or change to continue)",
                )
            else:
                # User switched to a new sign; reset lock
                self.state = GestureState.IDLE
                self.last_emitted_sign = None
                self.reset_hold()

        # Case 4: Candidate switched during holding
        if sign != self.current_candidate:
            self.current_candidate = sign
            self.hold_count = 1
            self.state = GestureState.HOLDING
        else:
            # Same candidate: increment hold counter
            self.hold_count += 1

        progress = min(self.hold_count / max(self.hold_frames_threshold, 1), 1.0)

        # Case 5: Threshold reached -> Commit character
        if self.hold_count >= self.hold_frames_threshold:
            self.state = GestureState.LOCKED
            self.last_emitted_sign = sign
            self.cooldown_count = self.cooldown_frames
            emitted = sign
            self.hold_count = 0

            return GestureStateOutput(
                state=GestureState.COMMITTED,
                active_sign=sign,
                hold_progress=1.0,
                emitted_char=emitted,
                status_message=f"Recognized: '{sign}'!",
            )

        # Case 6: Currently holding sign towards threshold
        self.state = GestureState.HOLDING
        return GestureStateOutput(
            state=GestureState.HOLDING,
            active_sign=sign,
            hold_progress=progress,
            emitted_char=None,
            status_message=f"Holding '{sign}' ({int(progress * 100)}%)",
        )

    def reset_hold(self) -> None:
        """Resets the hold counter and current candidate."""
        self.current_candidate = None
        self.hold_count = 0
        if self.state != GestureState.LOCKED:
            self.state = GestureState.IDLE

    def full_reset(self) -> None:
        """Completely resets the state machine."""
        self.state = GestureState.IDLE
        self.current_candidate = None
        self.hold_count = 0
        self.cooldown_count = 0
        self.last_emitted_sign = None
