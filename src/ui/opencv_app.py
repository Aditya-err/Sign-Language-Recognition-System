"""
Real-Time OpenCV HUD Application for Indian Sign Language Recognition V2.
========================================================================

Author: Senior Python Developer & Computer Vision Engineer (Mentoring MCA Fresher)
Module: sign_language_v2.src.ui.opencv_app

Technical Overview for Interviews:
----------------------------------
This script provides the interactive real-time user interface:
  1. Live Camera Pipeline: Captures frames, runs MediaPipe Hands pose estimation,
     normalizes 126-D coordinates, and evaluates the Deep MLP classifier.
  2. Temporal Stabilization: Prediction smoothing + Finite State Machine hold detection.
  3. Interactive Text Construction: Text builder assembling words letter-by-letter.
  4. HUD Information Architecture:
     - Top Banner: Predicted class, actual confidence %, Top 3 hypotheses, hand configuration.
     - Progress Bar: Visual hold duration indicator before committing character.
     - Bottom Banner: Assembled sentence text buffer + Keyboard shortcuts.
     - Corner HUD: Real hardware FPS counter.

Keyboard Controls:
  - 'c' or 'C': Clear sentence text buffer.
  - 'b' or 'B': Backspace (delete last character).
  - SPACE: Insert word space.
  - 'q' or 'Q' or ESC: Cleanly exit application.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Adjust sys.path to find packages from repository root
sys_path_root = Path(__file__).resolve().parents[2]
if str(sys_path_root) not in sys.path:
    sys.path.insert(0, str(sys_path_root))

from src.inference.gesture_state import GestureState, GestureStateManager
from src.inference.predictor import SignPredictor
from src.inference.smoothing import PredictionSmoother
from src.inference.text_builder import TextBuilder
from src.inference.webcam import WebcamManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("opencv_app")


def draw_hud(
    frame: np.ndarray,
    predicted_sign: str,
    confidence: float,
    top_k: list,
    hand_type: str,
    gesture_state_info,
    text_buffer: str,
    fps: float,
) -> np.ndarray:
    """
    Renders professional, clean HUD overlays on top of the webcam display frame.
    """
    h, w, _ = frame.shape
    overlay = frame.copy()

    # 1. Top Header Banner (Height: 65px)
    cv2.rectangle(overlay, (0, 0), (w, 65), (20, 20, 24), -1)
    cv2.line(overlay, (0, 65), (w, 65), (55, 55, 65), 1)

    # 2. Bottom Footer Banner (Height: 80px)
    cv2.rectangle(overlay, (0, h - 80), (w, h), (20, 20, 24), -1)
    cv2.line(overlay, (0, h - 80), (w, h - 80), (55, 55, 65), 1)

    # Alpha blend banners for subtle translucent styling
    alpha = 0.85
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # --- TOP BANNER CONTENT ---
    # Prediction badge
    if predicted_sign:
        sign_color = (0, 255, 120) if confidence >= 0.70 else (0, 200, 255)
        cv2.putText(
            frame,
            f"SIGN: {predicted_sign}",
            (15, 42),
            cv2.FONT_HERSHEY_DUPLEX,
            1.1,
            sign_color,
            2,
            cv2.LINE_AA,
        )
        conf_text = f"({confidence * 100:.1f}%)"
        cv2.putText(
            frame,
            conf_text,
            (160, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

        # Top-3 candidate hypotheses
        if len(top_k) >= 3:
            top_str = f"Top-3: {top_k[0][0]}({top_k[0][1]:.2f})  {top_k[1][0]}({top_k[1][1]:.2f})  {top_k[2][0]}({top_k[2][1]:.2f})"
            cv2.putText(
                frame,
                top_str,
                (260, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (170, 170, 180),
                1,
                cv2.LINE_AA,
            )
    else:
        cv2.putText(
            frame,
            "SIGN: --",
            (15, 42),
            cv2.FONT_HERSHEY_DUPLEX,
            1.0,
            (140, 140, 150),
            2,
            cv2.LINE_AA,
        )

    # Hand status tag
    hand_tag = hand_type.replace("_", " ").title()
    hand_color = (0, 255, 120) if hand_type == "dual" else (255, 180, 0) if hand_type != "none" else (120, 120, 120)
    cv2.putText(
        frame,
        f"Hands: {hand_tag}",
        (260, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        hand_color,
        1,
        cv2.LINE_AA,
    )

    # FPS tag (Top-Right)
    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(
        frame,
        fps_text,
        (w - 110, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (240, 240, 240),
        2,
        cv2.LINE_AA,
    )

    # --- PROGRESS BAR (Directly above footer) ---
    progress = gesture_state_info.hold_progress
    bar_y = h - 86
    bar_h = 6
    cv2.rectangle(frame, (0, bar_y), (w, bar_y + bar_h), (40, 40, 45), -1)
    if progress > 0.0:
        fill_w = int(w * progress)
        p_color = (0, 255, 120) if progress >= 1.0 else (0, 190, 255)
        cv2.rectangle(frame, (0, bar_y), (fill_w, bar_y + bar_h), p_color, -1)

    # --- FOOTER BANNER CONTENT ---
    # Status Message
    status_msg = gesture_state_info.status_message
    cv2.putText(
        frame,
        status_msg,
        (15, h - 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (180, 210, 255),
        1,
        cv2.LINE_AA,
    )

    # Assembled Text Display with Blinking Cursor
    cursor = "_" if int(time.time() * 2) % 2 == 0 else " "
    display_text = f"TEXT: {text_buffer}{cursor}"
    cv2.putText(
        frame,
        display_text,
        (15, h - 22),
        cv2.FONT_HERSHEY_DUPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Hotkey instructions
    hotkeys = "[C] Clear  [B] Backspace  [SPACE] Space  [Q] Exit"
    cv2.putText(
        frame,
        hotkeys,
        (w - 380, h - 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (140, 140, 150),
        1,
        cv2.LINE_AA,
    )

    return frame


def run_opencv_app(
    camera_index: int = 0,
    confidence_threshold: float = 0.70,
    hold_frames: int = 12,
    smoothing_window: int = 7,
) -> None:
    """
    Main real-time webcam sign recognition application loop.
    """
    logger.info("Initializing Indian Sign Language V2 Real-Time Application...")

    # Initialize modules
    predictor = SignPredictor()
    webcam = WebcamManager(camera_index=camera_index, mirror_display=True)

    if not webcam.is_ready:
        logger.error("Webcam not accessible. Application exiting.")
        return

    smoother = PredictionSmoother(
        window_size=smoothing_window,
        min_confidence=confidence_threshold,
    )
    state_manager = GestureStateManager(hold_frames_threshold=hold_frames)
    text_builder = TextBuilder()

    window_name = "Indian Sign Language (ISL) Recognition Engine V2"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    logger.info("Starting real-time video stream. Press 'Q' or ESC to exit.")

    fps = 0.0
    frame_count = 0
    start_time = time.time()

    try:
        while True:
            t_frame_start = time.perf_counter()

            # 1. Capture and extract landmarks (unflipped frame for MediaPipe, flipped frame for display)
            frame_res = webcam.process_next_frame(draw_landmarks=True)
            if not frame_res.success:
                logger.warning("Frame read unsuccessful, retrying...")
                time.sleep(0.01)
                continue

            # 2. Predict sign from 126-D normalized vector
            pred_res = predictor.predict(frame_res.feature_vector)

            # 3. Apply temporal smoothing
            smoothed = smoother.update(pred_res)

            # 4. Update gesture hold state machine
            state_output = state_manager.update(smoothed)

            # 5. Check if character was emitted by state machine
            if state_output.emitted_char:
                text_builder.add_character(state_output.emitted_char)
                logger.info("Committed Sign: '%s' -> Sentence: '%s'",
                            state_output.emitted_char, text_builder.get_text())

            # 6. Compute instantaneous and moving-average FPS
            frame_count += 1
            t_elapsed = time.perf_counter() - t_frame_start
            inst_fps = 1.0 / max(t_elapsed, 1e-4)
            fps = 0.9 * fps + 0.1 * inst_fps if fps > 0 else inst_fps

            # 7. Render HUD overlays
            active_label = smoothed.class_label if smoothed.class_label else pred_res.class_label
            active_conf = smoothed.confidence if smoothed.class_label else pred_res.confidence

            display = draw_hud(
                frame=frame_res.display_frame,
                predicted_sign=active_label,
                confidence=active_conf,
                top_k=pred_res.top_k,
                hand_type=pred_res.hand_type,
                gesture_state_info=state_output,
                text_buffer=text_builder.get_text(),
                fps=fps,
            )

            cv2.imshow(window_name, display)

            # 8. Handle Keyboard Shortcuts
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == ord("Q") or key == 27:  # 'q' or ESC
                logger.info("Exit requested by user.")
                break
            elif key == ord("c") or key == ord("C"):
                text_builder.clear()
                logger.info("Cleared text buffer.")
            elif key == ord("b") or key == ord("B"):
                text_builder.backspace()
                logger.info("Backspace pressed. Text: '%s'", text_builder.get_text())
            elif key == ord(" ") or key == 32:  # SPACE
                text_builder.add_space()
                logger.info("Space added. Text: '%s'", text_builder.get_text())

    except KeyboardInterrupt:
        logger.info("Application interrupted via keyboard.")
    finally:
        webcam.release()
        cv2.destroyAllWindows()
        total_time = time.time() - start_time
        logger.info("Application shutdown. Total runtime: %.1f s across %d frames (avg: %.1f FPS)",
                    total_time, frame_count, frame_count / max(total_time, 1e-4))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch Real-Time Indian Sign Language Recognition V2."
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--conf_thresh", type=float, default=0.70, help="Confidence threshold (default: 0.70)")
    parser.add_argument("--hold_frames", type=int, default=12, help="Hold frames to commit sign (default: 12)")
    parser.add_argument("--smooth_window", type=int, default=7, help="Smoothing window size (default: 7)")

    args = parser.parse_args()

    run_opencv_app(
        camera_index=args.camera,
        confidence_threshold=args.conf_thresh,
        hold_frames=args.hold_frames,
        smoothing_window=args.smooth_window,
    )


if __name__ == "__main__":
    main()
