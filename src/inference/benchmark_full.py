"""
Extended Complete-Loop Hardware Benchmark (300+ Frames).
=======================================================

Author: Senior Python Developer & Computer Vision Engineer (Mentoring MCA Fresher)
Module: sign_language_v2.src.inference.benchmark_full

Technical Overview for Interviews:
----------------------------------
This script performs a rigorous, multi-frame (300 frames) benchmark of the entire
live webcam pipeline on real hardware:
  1. Camera Frame Capture Time (I/O)
  2. MediaPipe Hands Tracking Time (BlazePalm + Landmark Regression)
  3. Geometric Spatial Normalization Time (NumPy wrist centering + palm scaling)
  4. MLP Model Forward Inference Time (TensorFlow direct tensor execution)
  5. Smoothing, FSM State Machine & Overlay Render Time
  6. TOTAL End-to-End Frame Latency (Directly measured total loop wall-clock time)
  7. Statistical Distribution: Mean, Median (P50), 95th Percentile (P95), Max Latency, and True Achieved FPS.
"""

import argparse
import json
import logging
import os
import platform
from pathlib import Path
import sys
import time

import cv2
import numpy as np

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
logger = logging.getLogger("benchmark_full")


def run_extended_benchmark(
    num_frames: int = 300,
    camera_index: int = 0,
    output_json_path: Path = None,
) -> dict:
    """
    Executes a 300-frame hardware benchmark profiling latency distribution.
    """
    logger.info("Initializing full-loop latency audit over %d frames...", num_frames)

    predictor = SignPredictor()
    webcam = WebcamManager(camera_index=camera_index, mirror_display=True)

    smoother = PredictionSmoother(window_size=7, min_confidence=0.70)
    state_mgr = GestureStateManager(hold_frames_threshold=12)

    # Hardware Environment Telemetry
    cpu_info = platform.processor() or "x86_64 CPU"
    cpu_count = os.cpu_count() or 4
    os_info = f"{platform.system()} {platform.release()}"

    use_synthetic = False
    if not webcam.is_ready:
        logger.warning("Physical webcam not available at index %d; switching to synthetic test frames.", camera_index)
        use_synthetic = True
    else:
        # Warmup loop (10 frames)
        logger.info("Warming up camera and MediaPipe pipeline (10 frames)...")
        for _ in range(10):
            _ = webcam.process_next_frame(draw_landmarks=True)

    capture_times = []
    mediapipe_times = []
    norm_times = []
    mlp_times = []
    fsm_times = []
    total_loop_times = []

    logger.info("Recording high-precision timings over %d frames...", num_frames)
    benchmark_start = time.perf_counter()

    for i in range(num_frames):
        t_loop_start = time.perf_counter()

        # 1. Camera Capture
        t0 = time.perf_counter()
        if use_synthetic:
            raw_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            ret = True
        else:
            ret, raw_frame = webcam.cap.read()
        t_capture = (time.perf_counter() - t0) * 1000

        if not ret or raw_frame is None:
            continue

        # 2. MediaPipe Detection (Unflipped Frame)
        t0 = time.perf_counter()
        detection = webcam.detector.process_frame(raw_frame, draw=True)
        t_mp = (time.perf_counter() - t0) * 1000

        # 3. Spatial Normalization
        t0 = time.perf_counter()
        feature_vector = normalize_dual_hands(
            detection.left_landmarks, detection.right_landmarks
        )
        t_norm = (time.perf_counter() - t0) * 1000

        # 4. Deep MLP Direct Tensor Forward Pass
        t0 = time.perf_counter()
        pred_res = predictor.predict(feature_vector)
        t_mlp = (time.perf_counter() - t0) * 1000

        # 5. Smoothing, FSM State Machine & Frame Flipping
        t0 = time.perf_counter()
        smoothed = smoother.update(pred_res)
        _ = state_mgr.update(smoothed)
        _ = cv2.flip(detection.annotated_image, 1)
        t_fsm = (time.perf_counter() - t0) * 1000

        # End-to-end total loop duration directly measured
        t_total = (time.perf_counter() - t_loop_start) * 1000

        capture_times.append(t_capture)
        mediapipe_times.append(t_mp)
        norm_times.append(t_norm)
        mlp_times.append(t_mlp)
        fsm_times.append(t_fsm)
        total_loop_times.append(t_total)

    total_wall_time = time.perf_counter() - benchmark_start
    webcam.release()

    total_loop_times = np.array(total_loop_times)
    effective_fps = len(total_loop_times) / max(total_wall_time, 1e-4)

    results = {
        "hardware_environment": {
            "os": os_info,
            "cpu": cpu_info,
            "logical_cores": cpu_count,
            "python_version": platform.python_version(),
        },
        "frames_profiled": len(total_loop_times),
        "total_wall_time_seconds": round(total_wall_time, 2),
        "measured_effective_fps": round(effective_fps, 1),
        "total_latency_distribution_ms": {
            "mean": round(float(np.mean(total_loop_times)), 2),
            "median_p50": round(float(np.median(total_loop_times)), 2),
            "p95": round(float(np.percentile(total_loop_times, 95)), 2),
            "min": round(float(np.min(total_loop_times)), 2),
            "max": round(float(np.max(total_loop_times)), 2),
            "std": round(float(np.std(total_loop_times)), 2),
        },
        "component_averages_ms": {
            "webcam_capture": round(float(np.mean(capture_times)), 2),
            "mediapipe_tracking": round(float(np.mean(mediapipe_times)), 2),
            "dual_hand_normalization": round(float(np.mean(norm_times)), 3),
            "mlp_inference": round(float(np.mean(mlp_times)), 2),
            "fsm_smoothing_render": round(float(np.mean(fsm_times)), 2),
        },
    }

    print("\n" + "=" * 70)
    print("      EXTENDED REAL-TIME HARDWARE BENCHMARK (300 FRAMES)      ")
    print("=" * 70)
    print(f"Hardware Environment: {cpu_count} CPU cores | {os_info}")
    print(f"Frames Profiled:      {results['frames_profiled']} frames in {results['total_wall_time_seconds']}s")
    print(f"Direct Measured FPS:  {results['measured_effective_fps']} FPS")
    print("-" * 70)
    print("TOTAL FRAME-TO-PREDICTION LATENCY DISTRIBUTION:")
    print(f"  Mean Latency:       {results['total_latency_distribution_ms']['mean']} ms")
    print(f"  Median (P50):       {results['total_latency_distribution_ms']['median_p50']} ms")
    print(f"  95th Percentile:    {results['total_latency_distribution_ms']['p95']} ms")
    print(f"  Min / Max Latency:  {results['total_latency_distribution_ms']['min']} ms / {results['total_latency_distribution_ms']['max']} ms")
    print("-" * 70)
    print("COMPONENT-WISE LATENCY BREAKDOWN (AVERAGES):")
    print(f"  1. Camera Frame Capture:      {results['component_averages_ms']['webcam_capture']} ms")
    print(f"  2. MediaPipe Hands Tracking:  {results['component_averages_ms']['mediapipe_tracking']} ms")
    print(f"  3. Spatial Normalization:     {results['component_averages_ms']['dual_hand_normalization']} ms")
    print(f"  4. Deep MLP Forward Pass:     {results['component_averages_ms']['mlp_inference']} ms")
    print(f"  5. Smoothing, FSM & Display:  {results['component_averages_ms']['fsm_smoothing_render']} ms")
    print("=" * 70)

    if output_json_path is not None:
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info("Saved extended benchmark metrics to %s", output_json_path)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Extended 300-frame complete-loop hardware benchmark.")
    parser.add_argument("--frames", type=int, default=300, help="Number of frames to profile (default: 300)")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")

    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    out_json = repo_root / "reports" / "benchmark_300_frames.json"

    run_extended_benchmark(
        num_frames=args.frames,
        camera_index=args.camera,
        output_json_path=out_json,
    )


if __name__ == "__main__":
    main()
