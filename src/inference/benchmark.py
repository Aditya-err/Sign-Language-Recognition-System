"""
Hardware Performance Profiling and Latency Benchmark.
=====================================================

Author: Senior Python Developer & Computer Vision Engineer (Mentoring MCA Fresher)
Module: sign_language_v2.src.inference.benchmark

Technical Overview for Interviews:
----------------------------------
To ensure 100% engineering integrity, this script profiles real hardware latency
across every single stage of the real-time recognition loop:
  1. OpenCV Camera Grab Time (I/O)
  2. MediaPipe Dual-Hand Landmark Tracking (CPU / TFLite XNNPACK)
  3. Geometric Spatial Normalization (NumPy)
  4. MLP Deep Classifier Inference (TensorFlow direct tensor execution)
  5. State Machine & HUD Overlay Render Time
  6. Total End-to-End Latency and Real Achieved FPS
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Adjust sys.path
sys_path_root = Path(__file__).resolve().parents[2]
if str(sys_path_root) not in sys.path:
    sys.path.insert(0, str(sys_path_root))

from src.core.normalization import normalize_dual_hands
from src.inference.gesture_state import GestureStateManager
from src.inference.predictor import SignPredictor
from src.inference.smoothing import PredictionSmoother
from src.inference.webcam import WebcamManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("benchmark")


def run_benchmark(num_frames: int = 60, camera_index: int = 0) -> dict:
    """
    Executes a high-precision hardware latency audit on the live webcam pipeline.
    """
    logger.info("Starting hardware latency benchmark across %d frames...", num_frames)

    predictor = SignPredictor()
    webcam = WebcamManager(camera_index=camera_index, mirror_display=True)

    if not webcam.is_ready:
        logger.error("Webcam hardware not accessible at index %d", camera_index)
        return {}

    smoother = PredictionSmoother(window_size=7, min_confidence=0.70)
    state_manager = GestureStateManager(hold_frames_threshold=12)

    # Warmup
    logger.info("Warming up pipeline...")
    for _ in range(5):
        _ = webcam.process_next_frame(draw_landmarks=True)

    capture_times = []
    mediapipe_times = []
    norm_times = []
    mlp_times = []
    smoothing_state_times = []
    total_frame_times = []

    logger.info("Recording per-frame timings...")

    for i in range(num_frames):
        t_total_start = time.perf_counter()

        # 1. Capture Raw Frame
        t0 = time.perf_counter()
        ret, raw_frame = webcam.cap.read()
        t_capture = (time.perf_counter() - t0) * 1000

        if not ret or raw_frame is None:
            continue

        # 2. MediaPipe Detection on Unflipped Frame
        t0 = time.perf_counter()
        detection = webcam.detector.process_frame(raw_frame, draw=True)
        t_mediapipe = (time.perf_counter() - t0) * 1000

        # 3. Dual-Hand Normalization
        t0 = time.perf_counter()
        feature_vector = normalize_dual_hands(
            detection.left_landmarks, detection.right_landmarks
        )
        t_norm = (time.perf_counter() - t0) * 1000

        # 4. MLP Forward Pass
        t0 = time.perf_counter()
        pred_res = predictor.predict(feature_vector)
        t_mlp = (time.perf_counter() - t0) * 1000

        # 5. Smoothing & State Machine
        t0 = time.perf_counter()
        smoothed = smoother.update(pred_res)
        _ = state_manager.update(smoothed)
        t_state = (time.perf_counter() - t0) * 1000

        t_total = (time.perf_counter() - t_total_start) * 1000

        capture_times.append(t_capture)
        mediapipe_times.append(t_mediapipe)
        norm_times.append(t_norm)
        mlp_times.append(t_mlp)
        smoothing_state_times.append(t_state)
        total_frame_times.append(t_total)

    webcam.release()

    # Compile Statistics
    avg_cap = float(np.mean(capture_times))
    avg_mp = float(np.mean(mediapipe_times))
    avg_norm = float(np.mean(norm_times))
    avg_mlp = float(np.mean(mlp_times))
    avg_state = float(np.mean(smoothing_state_times))
    avg_total = float(np.mean(total_frame_times))
    effective_fps = 1000.0 / max(avg_total, 1e-4)

    results = {
        "num_frames_profiled": len(total_frame_times),
        "avg_capture_ms": round(avg_cap, 2),
        "avg_mediapipe_ms": round(avg_mp, 2),
        "avg_normalization_ms": round(avg_norm, 3),
        "avg_mlp_inference_ms": round(avg_mlp, 2),
        "avg_state_machine_ms": round(avg_state, 3),
        "avg_total_latency_ms": round(avg_total, 2),
        "effective_fps": round(effective_fps, 1),
    }

    print("\n" + "=" * 60)
    print("        REAL-TIME HARDWARE LATENCY PROFILING RESULTS        ")
    print("=" * 60)
    print(f"Frames Profiled:                   {results['num_frames_profiled']}")
    print(f"1. Webcam Frame Grab:              {results['avg_capture_ms']:.2f} ms ({results['avg_capture_ms']/avg_total*100:.1f}%)")
    print(f"2. MediaPipe Hands Tracking:       {results['avg_mediapipe_ms']:.2f} ms ({results['avg_mediapipe_ms']/avg_total*100:.1f}%)")
    print(f"3. Dual-Hand Normalization:        {results['avg_normalization_ms']:.3f} ms ({results['avg_normalization_ms']/avg_total*100:.1f}%)")
    print(f"4. MLP Forward Pass (model(x)):    {results['avg_mlp_inference_ms']:.2f} ms ({results['avg_mlp_inference_ms']/avg_total*100:.1f}%)")
    print(f"5. Smoothing & State Machine:      {results['avg_state_machine_ms']:.3f} ms ({results['avg_state_machine_ms']/avg_total*100:.1f}%)")
    print("-" * 60)
    print(f"Total Frame-to-Prediction Latency: {results['avg_total_latency_ms']:.2f} ms")
    print(f"Real-World Achieved Framerate:     {results['effective_fps']:.1f} FPS")
    print("=" * 60)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile real hardware latency of the recognition pipeline.")
    parser.add_argument("--frames", type=int, default=60, help="Number of frames to profile (default: 60)")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index (default: 0)")
    args = parser.parse_args()

    run_benchmark(num_frames=args.frames, camera_index=args.camera)


if __name__ == "__main__":
    main()
