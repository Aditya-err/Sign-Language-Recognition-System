"""
Hand Detection Module using MediaPipe Hands.
===========================================

Author: Senior AI/ML Engineer (Mentoring MCA Fresher)
Module: sign_language_v2.src.core.hand_detector

Technical Overview for Interviews:
----------------------------------
MediaPipe Hands operates in a two-stage machine learning pipeline:
  1. Palm Detector: A lightweight single-shot detector (BlazePalm) that locates
     hands across the full video frame and estimates a bounding box.
  2. Hand Landmark Model: A precise 2.5D regression model operating on the cropped
     palm box to predict the 3D coordinates of all 21 joint landmarks.

Indian Sign Language (ISL) Design:
  Because ISL encompasses both single-handed signs (e.g. 'C', '1') and two-handed
  signs (e.g. 'A', 'B', 'Z'), this detector is configured with `max_num_hands=2`.
  
  Unlike legacy V1, this detector:
    - Never fabricates zero landmarks when hands are missing.
    - Explicitly classifies which hand is Left and which is Right.
    - Returns a typed HandDetectionResult so the downstream classifier only runs
      when genuine hands are present.
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple
import cv2
import numpy as np
import mediapipe as mp


@dataclass
class HandDetectionResult:
    """
    Structured data container for hand detection results.
    
    Attributes:
        detected (bool): True if at least one hand is detected.
        hands_count (int): Number of hands detected (0, 1, or 2).
        left_landmarks (Optional[np.ndarray]): Array of shape (21, 3) for Left hand, or None.
        right_landmarks (Optional[np.ndarray]): Array of shape (21, 3) for Right hand, or None.
        annotated_image (np.ndarray): Image with landmarks drawn for user visual feedback.
    """
    detected: bool
    hands_count: int
    left_landmarks: Optional[np.ndarray]
    right_landmarks: Optional[np.ndarray]
    annotated_image: np.ndarray


class HandDetector:
    """
    Robust wrapper around MediaPipe Hands for Real-Time ISL detection.
    """

    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        """
        Initializes the MediaPipe Hands detector.
        
        Args:
            static_image_mode: False for video streams (uses tracking); True for static images.
            max_num_hands: Maximum hands to detect (2 for Indian Sign Language).
            min_detection_confidence: Minimum palm detection confidence [0.0, 1.0].
            min_tracking_confidence: Minimum landmark tracking confidence [0.0, 1.0].
        """
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process_frame(
        self, frame_bgr: np.ndarray, draw: bool = True
    ) -> HandDetectionResult:
        """
        Processes a single BGR video frame and extracts 3D hand landmarks.
        
        Args:
            frame_bgr: Standard OpenCV BGR frame (H, W, 3).
            draw: Whether to render landmark skeletons onto the annotated image.
            
        Returns:
            HandDetectionResult: Container with separated Left/Right landmarks and metadata.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return HandDetectionResult(
                detected=False,
                hands_count=0,
                left_landmarks=None,
                right_landmarks=None,
                annotated_image=np.zeros((100, 100, 3), dtype=np.uint8) if frame_bgr is None else frame_bgr,
            )

        # OpenCV uses BGR; MediaPipe expects RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        # Optimize performance by marking frame as non-writeable during inference
        frame_rgb.flags.writeable = False
        results = self.hands.process(frame_rgb)
        frame_rgb.flags.writeable = True

        annotated_image = frame_bgr.copy() if draw else frame_bgr

        left_hand_coords: Optional[np.ndarray] = None
        right_hand_coords: Optional[np.ndarray] = None
        detected_count = 0

        if results.multi_hand_landmarks and results.multi_handedness:
            detected_count = len(results.multi_hand_landmarks)

            for hand_landmarks, handedness_info in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                # MediaPipe classifies as 'Left' or 'Right'
                # Note: MediaPipe assumes camera is mirrored by default; label is accurate for user
                label = handedness_info.classification[0].label  # 'Left' or 'Right'

                # Extract (x, y, z) coordinates into an array of shape (21, 3)
                coords = np.array(
                    [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
                    dtype=np.float32,
                )

                if label == "Left":
                    left_hand_coords = coords
                elif label == "Right":
                    right_hand_coords = coords

                # Draw landmark skeleton if requested
                if draw:
                    self.mp_drawing.draw_landmarks(
                        annotated_image,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style(),
                    )

        return HandDetectionResult(
            detected=(detected_count > 0),
            hands_count=detected_count,
            left_landmarks=left_hand_coords,
            right_landmarks=right_hand_coords,
            annotated_image=annotated_image,
        )

    def close(self) -> None:
        """Releases native MediaPipe resources."""
        if hasattr(self, "hands") and self.hands:
            self.hands.close()
