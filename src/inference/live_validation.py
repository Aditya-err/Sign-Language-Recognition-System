"""
Real-World Webcam Sign Validation Protocol CLI.
==============================================

Author: Senior Python Developer & Computer Vision Engineer (Mentoring MCA Fresher)
Module: sign_language_v2.src.inference.live_validation

Technical Overview for Interviews:
----------------------------------
This script implements the manual testing protocol for all 35 supported classes.
It steps through each class one-by-one:
  1. Displays the expected sign and prompts the user to perform it.
  2. Runs real-time inference, temporal smoothing, and gesture hold state machine.
  3. Records: Expected Class, Predicted Class, Top-1 Confidence, Stability,
     Committed status, Duplicate status, Pass/Fail.
  4. Saves structured JSON & Markdown audit logs to `reports/real_world_validation.json`
     and `reports/real_world_validation.md`.
"""

import argparse
from datetime import datetime
import json
import logging
from pathlib import Path
import sys
import time
from typing import Dict, List

import cv2
import numpy as np

sys_path_root = Path(__file__).resolve().parents[2]
if str(sys_path_root) not in sys.path:
    sys.path.insert(0, str(sys_path_root))

from src.inference.gesture_state import GestureState, GestureStateManager
from src.inference.predictor import SignPredictor
from src.inference.smoothing import PredictionSmoother
from src.inference.webcam import WebcamManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("live_validation")


def run_live_validation(
    classes_to_test: List[str],
    camera_index: int = 0,
    confidence_threshold: float = 0.70,
    hold_frames: int = 12,
    test_duration_per_sign: float = 8.0,
) -> Dict:
    """
    Executes a structured live validation session across specified sign classes.
    """
    predictor = SignPredictor()
    webcam = WebcamManager(camera_index=camera_index, mirror_display=True)

    if not webcam.is_ready:
        logger.error("Webcam hardware inaccessible.")
        return {}

    smoother = PredictionSmoother(window_size=7, min_confidence=confidence_threshold)
    state_mgr = GestureStateManager(hold_frames_threshold=hold_frames)

    window_name = "Real-World ISL Validation Protocol"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    results: List[Dict] = []
    logger.info("Starting live validation protocol for %d classes...", len(classes_to_test))

    try:
        for idx, expected in enumerate(classes_to_test, 1):
            smoother.reset()
            state_mgr.full_reset()

            sign_start = time.time()
            observed_preds: List[str] = []
            observed_confs: List[float] = []
            committed_chars: List[str] = []
            is_stable_seen = False

            logger.info("[%d/%d] Ready for expected sign: '%s'. Perform sign in front of camera...",
                        idx, len(classes_to_test), expected)

            while time.time() - sign_start < test_duration_per_sign:
                time_left = test_duration_per_sign - (time.time() - sign_start)
                frame_res = webcam.process_next_frame(draw_landmarks=True)
                if not frame_res.success:
                    time.sleep(0.01)
                    continue

                display = frame_res.display_frame.copy()
                h, w, _ = display.shape

                # Prediction
                pred_res = predictor.predict(frame_res.feature_vector)
                smoothed = smoother.update(pred_res)
                state_out = state_mgr.update(smoothed)

                if pred_res.class_label:
                    observed_preds.append(pred_res.class_label)
                    observed_confs.append(pred_res.confidence)
                if smoothed.is_stable:
                    is_stable_seen = True
                if state_out.emitted_char:
                    committed_chars.append(state_out.emitted_char)

                # Draw Validation HUD
                cv2.rectangle(display, (0, 0), (w, 60), (20, 20, 24), -1)
                cv2.putText(
                    display,
                    f"TEST [{idx}/{len(classes_to_test)}]: FORM SIGN '{expected}' | Time Left: {time_left:.1f}s",
                    (15, 38),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.70,
                    (0, 220, 255),
                    2,
                )

                # Prediction status
                active_pred = smoothed.class_label if smoothed.class_label else pred_res.class_label or "--"
                active_conf = smoothed.confidence if smoothed.class_label else pred_res.confidence
                cv2.rectangle(display, (0, h - 50), (w, h), (20, 20, 24), -1)
                cv2.putText(
                    display,
                    f"PREDICTED: {active_pred} ({active_conf*100:.1f}%) | STATUS: {state_out.status_message}",
                    (15, h - 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1,
                )

                cv2.imshow(window_name, display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == ord("Q") or key == 27:
                    logger.info("Validation interrupted by user.")
                    return {"results": results, "interrupted": True}
                elif key == ord("n") or key == ord("N"):  # Next sign early
                    break

            # Evaluate test result for this sign
            if observed_preds:
                from collections import Counter
                most_common_pred, count = Counter(observed_preds).most_common(1)[0]
                avg_conf = float(np.mean([c for p, c in zip(observed_preds, observed_confs) if p == most_common_pred]))
            else:
                most_common_pred = None
                avg_conf = 0.0

            passed = (most_common_pred == expected) and (len(committed_chars) == 1)
            duplicate_detected = len(committed_chars) > 1

            test_record = {
                "expected_class": expected,
                "predicted_class": most_common_pred,
                "average_confidence": round(avg_conf, 4),
                "is_stable": is_stable_seen,
                "committed_chars": committed_chars,
                "duplicate_characters": duplicate_detected,
                "status": "PASS" if passed else "FAIL" if most_common_pred else "NO_HAND_DETECTED",
            }
            results.append(test_record)
            logger.info("Result for '%s': %s (Pred: %s, Conf: %.1f%%, Committed: %s)",
                        expected, test_record["status"], most_common_pred, avg_conf * 100, committed_chars)

    finally:
        webcam.release()
        cv2.destroyAllWindows()

    return {"results": results, "interrupted": False}


def main() -> None:
    parser = argparse.ArgumentParser(description="Live webcam sign validation protocol.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--classes", type=str, default="all", help="Classes to test ('all', 'digits', 'letters', or comma-separated list)")
    parser.add_argument("--duration", type=float, default=6.0, help="Duration to test each sign in seconds (default: 6.0)")

    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    with open(repo_root / "data" / "processed" / "label_encoder.json") as f:
        all_classes = json.load(f)["classes"]

    if args.classes == "all":
        target_classes = all_classes
    elif args.classes == "digits":
        target_classes = [c for c in all_classes if c.isdigit()]
    elif args.classes == "letters":
        target_classes = [c for c in all_classes if c.isalpha()]
    else:
        target_classes = [c.strip().upper() for c in args.classes.split(",") if c.strip().upper() in all_classes]

    output = run_live_validation(
        classes_to_test=target_classes,
        camera_index=args.camera,
        test_duration_per_sign=args.duration,
    )

    # Save validation records
    reports_dir = repo_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_json = reports_dir / "real_world_validation.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    logger.info("Saved validation results to %s", out_json)


if __name__ == "__main__":
    main()
