"""
Webcam Hand Detection and Frame Acquisition Integration.
========================================================

Author: Senior Python Developer & Computer Vision Engineer (Mentoring MCA Fresher)
Module: sign_language_v2.src.inference.webcam

Technical Overview for Interviews (Critical Handedness Finding):
----------------------------------------------------------------
MediaPipe Hands internally assumes the input image is a front-facing mirrored selfie feed.
In our Phase 2.1 Technical Audit, we identified a critical failure mode:
  - In `prepare_dataset.py`, archive images were unmirrored third-person photographs.
    MediaPipe labeled the signer's right hand as 'Right' (stored in features[63:126]).
  - If a webcam app flips the frame horizontally with `cv2.flip(frame, 1)` BEFORE
    MediaPipe processes it, MediaPipe inverts the handedness label to 'Left' (features[0:63]),
    causing prediction accuracy on right-hand signs to collapse (e.g. '1' -> 'V').

The Architectural Solution:
  1. Capture raw, unflipped webcam frame from OpenCV.
  2. Pass the UNFLIPPED frame to MediaPipe Hands for feature extraction and skeleton rendering.
     This guarantees handedness slots (Left: 0-62, Right: 63-125) match the training dataset.
  3. Flip the resulting annotated frame horizontally ONLY for visual display so the user
     experiences an intuitive mirror reflection.
"""

from dataclasses import dataclass
import logging
from typing import Optional, Tuple

import cv2
import numpy as np

from src.core.hand_detector import HandDetectionResult, HandDetector
from src.core.normalization import normalize_dual_hands

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("webcam")


@dataclass
class FrameProcessingResult:
    """
    Container for webcam frame processing outputs.

    Attributes:
        success (bool): True if frame was successfully captured and processed.
        display_frame (np.ndarray): Mirrored frame with landmark skeleton overlay for HUD display.
        feature_vector (np.ndarray): 126-D normalized feature vector matching model training space.
        detection (HandDetectionResult): Structured MediaPipe detection result.
    """
    success: bool
    display_frame: np.ndarray
    feature_vector: np.ndarray
    detection: HandDetectionResult


class WebcamManager:
    """
    Manages OpenCV video capture, MediaPipe hand detection, and handedness-aligned normalization.
    """

    def __init__(
        self,
        camera_index: int = 0,
        frame_width: int = 640,
        frame_height: int = 480,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.5,
        mirror_display: bool = True,
    ) -> None:
        """
        Initializes the webcam capture and MediaPipe detector.

        Args:
            camera_index: OpenCV camera hardware index (default: 0).
            frame_width: Requested frame width in pixels.
            frame_height: Requested frame height in pixels.
            min_detection_confidence: MediaPipe palm detection threshold.
            min_tracking_confidence: MediaPipe landmark tracking threshold.
            mirror_display: Whether to flip the visual HUD display horizontally.
        """
        self.camera_index = camera_index
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.mirror_display = mirror_display

        logger.info("Initializing webcam at device index %d...", camera_index)
        self.cap = cv2.VideoCapture(camera_index)

        if not self.cap.isOpened():
            logger.error("Failed to open webcam at index %d", camera_index)
            self.is_ready = False
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info("Webcam initialized successfully (%dx%d)", actual_w, actual_h)

        # Initialize MediaPipe Hands in streaming mode (static_image_mode=False for fast video tracking)
        self.detector = HandDetector(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.is_ready = True

    def process_next_frame(self, draw_landmarks: bool = True) -> FrameProcessingResult:
        """
        Captures one webcam frame, extracts 126-D landmarks, and prepares visual display.

        Returns:
            FrameProcessingResult container.
        """
        if not self.is_ready or not self.cap.isOpened():
            dummy = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
            return FrameProcessingResult(
                success=False,
                display_frame=dummy,
                feature_vector=np.zeros(126, dtype=np.float32),
                detection=HandDetectionResult(False, 0, None, None, dummy),
            )

        ret, raw_frame = self.cap.read()
        if not ret or raw_frame is None or raw_frame.size == 0:
            logger.warning("Failed to grab frame from webcam")
            dummy = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
            return FrameProcessingResult(
                success=False,
                display_frame=dummy,
                feature_vector=np.zeros(126, dtype=np.float32),
                detection=HandDetectionResult(False, 0, None, None, dummy),
            )

        # CRITICAL HANDEDNESS RESOLUTION:
        # Feed the UNFLIPPED frame to MediaPipe to ensure handedness matches training data
        detection = self.detector.process_frame(raw_frame, draw=draw_landmarks)

        # Extract normalized 126-D dual hand feature vector (Left: 0..62, Right: 63..125)
        feature_vector = normalize_dual_hands(
            left_landmarks=detection.left_landmarks,
            right_landmarks=detection.right_landmarks,
        )

        # Prepare visual display frame (flip horizontally if mirror_display is enabled)
        if self.mirror_display:
            display_frame = cv2.flip(detection.annotated_image, 1)
        else:
            display_frame = detection.annotated_image

        return FrameProcessingResult(
            success=True,
            display_frame=display_frame,
            feature_vector=feature_vector,
            detection=detection,
        )

    def release(self) -> None:
        """Safely releases OpenCV camera hardware and MediaPipe resources."""
        if hasattr(self, "cap") and self.cap.isOpened():
            self.cap.release()
            logger.info("Webcam hardware released")
        if hasattr(self, "detector") and self.detector:
            self.detector.close()
            logger.info("MediaPipe detector released")
        self.is_ready = False
