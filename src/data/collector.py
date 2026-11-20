"""
Guided Webcam Data Collector for Indian Sign Language (ISL) Recognition.
=======================================================================

Author: Senior AI/ML Engineer (Mentoring MCA Fresher)
Module: sign_language_v2.src.data.collector

Technical Overview for Interviews:
----------------------------------
To prevent data leakage and ensure real-world generalization, every collected
sample must be tagged with explicit provenance metadata (Participant ID,
Session ID, Timestamp, Handedness Configuration).

Key Design Principles:
  1. No Duplicated Frames: In contrast to legacy V1 (which duplicated single frames 30x),
     this collector captures genuine spatial snapshots of static hand signs.
  2. Dual-Hand Support: Natively supports both single-handed (e.g., '1', 'C') and
     two-handed (e.g., 'A', 'B', 'Z') Indian Sign Language signs.
  3. Strict Metadata Preservation: Saved JSON samples retain both raw coordinates
     and normalized 126-D vectors for complete auditability.
"""

import os
import sys
import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.hand_detector import HandDetector, HandDetectionResult
from src.core.normalization import normalize_dual_hands


class GuidedDataCollector:
    """
    Interactive OpenCV data collection tool with real-time HUD feedback.
    """

    def __init__(
        self,
        output_dir: str = "data/raw",
        camera_index: int = 0,
    ) -> None:
        """
        Initializes the collector.
        
        Args:
            output_dir: Directory where raw JSON sample records are saved.
            camera_index: OpenCV camera hardware index (default 0).
        """
        self.output_base = PROJECT_ROOT / output_dir
        self.camera_index = camera_index
        self.detector = HandDetector(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        )

    def _determine_hand_config(self, result: HandDetectionResult) -> str:
        """Determines anatomical configuration of detected hands."""
        if result.left_landmarks is not None and result.right_landmarks is not None:
            return "two_hand"
        elif result.left_landmarks is not None:
            return "single_left"
        elif result.right_landmarks is not None:
            return "single_right"
        return "none"

    def save_sample(
        self,
        label: str,
        participant_id: str,
        session_id: str,
        detection: HandDetectionResult,
    ) -> Optional[Path]:
        """
        Serializes a genuine landmark sample with full provenance metadata.
        
        Args:
            label: Gesture sign class (e.g., 'A', 'B', '1').
            participant_id: Unique human participant identifier (e.g., 'P01').
            session_id: Collection session ID (e.g., 'S01').
            detection: HandDetectionResult containing detected coordinates.
            
        Returns:
            Path to the saved JSON file, or None if no hand was detected.
        """
        if not detection.detected:
            return None

        hand_config = self._determine_hand_config(detection)
        
        # Compute normalized 126-D feature vector (Left: 0-62, Right: 63-125)
        normalized_features = normalize_dual_hands(
            left_landmarks=detection.left_landmarks,
            right_landmarks=detection.right_landmarks,
        ).tolist()

        timestamp = datetime.utcnow().isoformat() + "Z"
        sample_id = str(uuid.uuid4())[:8]

        sample_record: Dict[str, Any] = {
            "version": "2.0",
            "sample_id": sample_id,
            "label": label.upper(),
            "participant_id": participant_id.upper(),
            "session_id": session_id.upper(),
            "timestamp": timestamp,
            "hand_config": hand_config,
            "hands_count": detection.hands_count,
            "raw_landmarks": {
                "left_hand": detection.left_landmarks.tolist() if detection.left_landmarks is not None else None,
                "right_hand": detection.right_landmarks.tolist() if detection.right_landmarks is not None else None,
            },
            "normalized_features": normalized_features,
        }

        # Target directory: data/raw/<LABEL>/
        class_dir = self.output_base / label.upper()
        class_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{label.upper()}_{participant_id.upper()}_{session_id.upper()}_{sample_id}.json"
        target_path = class_dir / filename

        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(sample_record, f, indent=2)

        return target_path

    def run_interactive(
        self,
        label: str,
        participant_id: str,
        session_id: str,
        target_samples: int = 50,
    ) -> int:
        """
        Launches the real-time webcam collection interface.
        
        Controls:
          [SPACE] : Capture single clean landmark sample
          [R]     : Toggle continuous burst recording mode (every 250ms)
          [Q]     : Exit collection session cleanly
        """
        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            # Fallback for systems without CAP_DSHOW
            cap = cv2.VideoCapture(self.camera_index)
            
        if not cap.isOpened():
            print(f"[ERROR] Could not open webcam at index {self.camera_index}.")
            return 0

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        collected_count = 0
        burst_mode = False
        last_burst_time = 0.0
        burst_interval = 0.25  # 250ms interval between burst frames

        print("\n" + "=" * 60)
        print(f" Guided ISL Data Collection | Target: '{label.upper()}'")
        print(f" Participant: {participant_id.upper()} | Session: {session_id.upper()}")
        print(" Controls: [SPACE] Single Capture | [R] Toggle Burst | [Q] Exit")
        print("=" * 60 + "\n")

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("[WARNING] Failed to grab webcam frame.")
                    break

                # Flip frame horizontally for intuitive selfie mirror display
                frame = cv2.flip(frame, 1)

                # Process frame for MediaPipe landmarks
                detection = self.detector.process_frame(frame, draw=True)
                display_frame = detection.annotated_image

                # Check burst recording trigger
                current_time = time.time()
                if burst_mode and detection.detected:
                    if current_time - last_burst_time >= burst_interval:
                        saved_path = self.save_sample(
                            label, participant_id, session_id, detection
                        )
                        if saved_path:
                            collected_count += 1
                            last_burst_time = current_time

                # Draw Professional HUD
                # Top Header Banner
                cv2.rectangle(display_frame, (0, 0), (640, 50), (20, 20, 20), -1)
                cv2.line(display_frame, (0, 50), (640, 50), (60, 60, 60), 1)

                # Class & Participant Info
                status_color = (0, 255, 120) if detection.detected else (0, 100, 255)
                hand_text = self._determine_hand_config(detection).upper()
                cv2.putText(
                    display_frame,
                    f"TARGET: {label.upper()} | P_ID: {participant_id.upper()} | S_ID: {session_id.upper()}",
                    (12, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                # Bottom Footer Banner
                cv2.rectangle(display_frame, (0, 430), (640, 480), (20, 20, 20), -1)
                cv2.line(display_frame, (0, 430), (640, 430), (60, 60, 60), 1)

                cv2.putText(
                    display_frame,
                    f"Samples: {collected_count}/{target_samples} | Hands: {hand_text} ({detection.hands_count})",
                    (12, 460),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    status_color,
                    2,
                    cv2.LINE_AA,
                )

                # Recording Indicator
                if burst_mode:
                    cv2.circle(display_frame, (615, 25), 10, (0, 0, 255), -1)
                    cv2.putText(
                        display_frame,
                        "REC",
                        (565, 31),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )

                cv2.imshow("ISL Guided Data Collector V2", display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == 32:  # Spacebar single capture
                    if detection.detected:
                        saved_path = self.save_sample(
                            label, participant_id, session_id, detection
                        )
                        if saved_path:
                            collected_count += 1
                            # Visual flash confirmation
                            cv2.rectangle(display_frame, (0, 0), (640, 480), (255, 255, 255), 4)
                            cv2.imshow("ISL Guided Data Collector V2", display_frame)
                            cv2.waitKey(40)
                    else:
                        print("[INFO] No hands in frame to capture. Please position your hand.")
                elif key == ord("r"):
                    burst_mode = not burst_mode
                    print(f"[INFO] Burst recording mode: {'ACTIVE' if burst_mode else 'STOPPED'}")

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.detector.close()

        print(f"\n[SUMMARY] Session complete. Captured {collected_count} valid samples for '{label.upper()}'.")
        return collected_count


def main():
    """Interactive CLI runner for data collection."""
    print("=" * 60)
    print("   INDIAN SIGN LANGUAGE (ISL) V2 — GUIDED DATA COLLECTOR")
    print("=" * 60)
    
    label = input("Enter Gesture Label (e.g. A, B, 1, 2): ").strip()
    if not label:
        print("[ERROR] Label cannot be empty.")
        return

    participant_id = input("Enter Participant ID (e.g. P01, P02): ").strip() or "P01"
    session_id = input("Enter Session ID (e.g. S01, S02): ").strip() or "S01"
    
    try:
        target_samples = int(input("Target samples to collect [default: 50]: ").strip() or "50")
    except ValueError:
        target_samples = 50

    collector = GuidedDataCollector()
    collector.run_interactive(
        label=label,
        participant_id=participant_id,
        session_id=session_id,
        target_samples=target_samples,
    )


if __name__ == "__main__":
    main()
