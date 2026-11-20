"""
Independent Real-World Live Data Recorder.
==========================================

Author: Senior Python Developer & Computer Vision Engineer (Mentoring MCA Fresher)
Module: sign_language_v2.src.data.live_recorder

Technical Overview for Interviews:
----------------------------------
To perform rigorous, non-leaking real-world evaluation, models must be benchmarked
on genuine webcam recordings distinct from the original training archive.

This module provides a structured collection tool for independent real-world sessions:
  - Participant ID Tracking: Identifies human signers (e.g. 'P01', 'P02_UNSEEN').
  - Session ID & Environmental Tagging: Records lighting ('normal', 'bright', 'low_light')
    and camera distance ('near', 'medium', 'far').
  - Provenance Isolation: Saves data to `data/real_world/` rather than mixing into
    `data/processed/`, preventing data contamination.
  - Full Kinematic Storage: Serializes both raw MediaPipe 3D coordinates and the
    normalized 126-D feature vector with ISO timestamps.
"""

import argparse
from datetime import datetime
import json
import logging
from pathlib import Path
import sys
import time
import uuid

import cv2
import numpy as np

sys_path_root = Path(__file__).resolve().parents[2]
if str(sys_path_root) not in sys.path:
    sys.path.insert(0, str(sys_path_root))

from src.core.normalization import normalize_dual_hands
from src.inference.webcam import WebcamManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("live_recorder")


class RealWorldRecorder:
    """
    Collects and serializes independent webcam gesture samples with environmental metadata.
    """

    def __init__(
        self,
        output_dir: Path,
        participant_id: str = "P01",
        session_id: str = "S01",
        condition: str = "normal_indoor",
    ) -> None:
        self.output_dir = output_dir
        self.participant_id = participant_id.upper()
        self.session_id = session_id.upper()
        self.condition = condition

        self.session_dir = self.output_dir / self.participant_id / self.session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Real-world session storage initialized at: %s", self.session_dir)

    def record_sample(
        self,
        label: str,
        detection,
        feature_vector: np.ndarray,
    ) -> Optional[Path]:
        """
        Saves a single gesture sample with environmental metadata.
        """
        if not detection.detected:
            return None

        clean_label = label.strip().upper()
        class_dir = self.session_dir / clean_label
        class_dir.mkdir(parents=True, exist_ok=True)

        sample_id = str(uuid.uuid4())[:8]
        timestamp = datetime.utcnow().isoformat() + "Z"

        record = {
            "version": "2.1",
            "sample_id": sample_id,
            "label": clean_label,
            "participant_id": self.participant_id,
            "session_id": self.session_id,
            "condition": self.condition,
            "timestamp": timestamp,
            "hands_count": detection.hands_count,
            "has_left": detection.left_landmarks is not None,
            "has_right": detection.right_landmarks is not None,
            "raw_left_landmarks": (
                detection.left_landmarks.tolist()
                if detection.left_landmarks is not None
                else None
            ),
            "raw_right_landmarks": (
                detection.right_landmarks.tolist()
                if detection.right_landmarks is not None
                else None
            ),
            "feature_vector_126": feature_vector.tolist(),
        }

        file_path = class_dir / f"sample_{sample_id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)

        return file_path


def run_collection_session(
    target_class: str,
    target_samples: int = 15,
    camera_index: int = 0,
    participant_id: str = "P01",
    session_id: str = "S01",
    condition: str = "normal_indoor",
) -> int:
    """
    Interactive collection session for a specific sign class.
    """
    repo_root = Path(__file__).resolve().parents[2]
    out_dir = repo_root / "data" / "real_world"

    recorder = RealWorldRecorder(
        output_dir=out_dir,
        participant_id=participant_id,
        session_id=session_id,
        condition=condition,
    )
    webcam = WebcamManager(camera_index=camera_index, mirror_display=True)

    if not webcam.is_ready:
        logger.error("Webcam hardware not available.")
        return 0

    window_name = f"Real-World Recorder — Sign: {target_class.upper()}"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    collected = 0
    last_record_time = 0.0
    record_interval = 0.35  # seconds between bursts

    logger.info("Form sign '%s'. Press SPACE to record bursts, 'Q' to finish.", target_class.upper())

    try:
        while collected < target_samples:
            frame_res = webcam.process_next_frame(draw_landmarks=True)
            if not frame_res.success:
                time.sleep(0.01)
                continue

            display = frame_res.display_frame.copy()
            h, w, _ = display.shape

            # Header HUD
            cv2.rectangle(display, (0, 0), (w, 55), (20, 20, 24), -1)
            cv2.putText(
                display,
                f"TARGET: {target_class.upper()} | COLLECTED: {collected}/{target_samples} | PARTICIPANT: {participant_id}",
                (15, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 120) if frame_res.detection.detected else (0, 100, 255),
                2,
            )

            # Footer HUD
            cv2.rectangle(display, (0, h - 45), (w, h), (20, 20, 24), -1)
            cv2.putText(
                display,
                "[SPACE] Record Burst   [Q] Finish / Skip Class",
                (15, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (200, 200, 200),
                1,
            )

            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == ord("Q") or key == 27:
                logger.info("Collection session stopped by user.")
                break
            elif key == ord(" ") or key == 32:
                # Burst recording
                now = time.time()
                if now - last_record_time >= record_interval and frame_res.detection.detected:
                    saved_p = recorder.record_sample(
                        label=target_class,
                        detection=frame_res.detection,
                        feature_vector=frame_res.feature_vector,
                    )
                    if saved_p:
                        collected += 1
                        last_record_time = now
                        logger.info("Recorded sample %d/%d for '%s'", collected, target_samples, target_class)

    finally:
        webcam.release()
        cv2.destroyAllWindows()

    return collected


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect independent real-world webcam samples.")
    parser.add_argument("--sign", type=str, required=True, help="Target sign label (e.g. 'A', '1')")
    parser.add_argument("--samples", type=int, default=15, help="Number of samples to collect (default: 15)")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("--participant", type=str, default="P01", help="Participant ID (e.g. P01)")
    parser.add_argument("--session", type=str, default="S01", help="Session ID (e.g. S01)")
    parser.add_argument("--condition", type=str, default="normal_indoor", help="Environmental condition tag")

    args = parser.parse_args()
    run_collection_session(
        target_class=args.sign,
        target_samples=args.samples,
        camera_index=args.camera,
        participant_id=args.participant,
        session_id=args.session,
        condition=args.condition,
    )


if __name__ == "__main__":
    main()
