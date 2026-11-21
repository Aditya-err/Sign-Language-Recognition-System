"""
Modern Glassmorphism Web & Desktop GUI for Indian Sign Language Recognition V2.
================================================================================

Module: sign_language_v2.src.ui.web_app
Author: Senior AI/ML Engineer & Python Developer

This module delivers a high-performance web-based desktop application utilizing
native HTML5/CSS3 glassmorphism, responsive video streaming, and pywebview.
It requires zero external web frameworks, using Python's standard library
ThreadingHTTPServer to stream MJPEG frames and real-time telemetry.
"""

import base64
import json
import logging
import socket
import sys
import threading
import time
import webbrowser
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Ensure project root is in sys.path
sys_path_root = Path(__file__).resolve().parents[2]
if str(sys_path_root) not in sys.path:
    sys.path.insert(0, str(sys_path_root))

from src.core.normalization import normalize_dual_hands
from src.inference.gesture_state import GestureState, GestureStateManager
from src.inference.predictor import SignPredictor
from src.inference.smoothing import PredictionSmoother
from src.inference.text_builder import TextBuilder
from src.inference.webcam import WebcamManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("web_app")


class ISLWebService:
    """
    Core backend service managing computer vision, deep learning inference,
    state machines, and MJPEG video streaming.
    """

    def __init__(
        self,
        camera_index: int = 0,
        confidence_threshold: float = 0.70,
        hold_frames: int = 12,
        smoothing_window: int = 7,
    ):
        self.camera_index = camera_index
        self.confidence_threshold = confidence_threshold
        self.hold_frames = hold_frames
        self.smoothing_window = smoothing_window

        self.running = threading.Event()
        self.paused = False
        self.show_mesh = True
        self.lock = threading.Lock()

        # Latest JPEG frame bytes for streaming
        self.latest_jpeg: Optional[bytes] = None

        # Telemetry payload
        self.telemetry = {
            "fps": 0.0,
            "hands_detected": 0,
            "sign": None,
            "confidence": 0.0,
            "raw_sign": None,
            "top_3": [],
            "state": "IDLE",
            "hold_progress": 0.0,
            "current_word": "",
            "sentence": "",
            "image": "",
        }

        # Initialize text builder
        self.text_builder = TextBuilder()

    def start_vision_pipeline(self):
        self.running.set()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def stop_vision_pipeline(self):
        self.running.clear()
        if hasattr(self, "thread"):
            self.thread.join(timeout=1.5)

    def _run_loop(self):
        logger.info("Initializing Vision & Inference Engine...")
        predictor = SignPredictor()
        webcam = WebcamManager(camera_index=self.camera_index, mirror_display=False)

        if not webcam.is_ready:
            logger.error("Webcam not ready at index %d", self.camera_index)
            return

        smoother = PredictionSmoother(
            window_size=self.smoothing_window,
            min_confidence=self.confidence_threshold,
        )
        state_mgr = GestureStateManager(hold_frames_threshold=self.hold_frames)

        fps_timer = time.perf_counter()
        fps_counter = 0
        current_fps = 0.0

        try:
            while self.running.is_set():
                if self.paused:
                    time.sleep(0.05)
                    continue

                # Safely sync controls
                with self.lock:
                    smoother.min_confidence = self.confidence_threshold
                    state_mgr.hold_frames_threshold = self.hold_frames
                    draw_mesh = self.show_mesh

                ret, raw_frame = webcam.cap.read()
                if not ret or raw_frame is None:
                    time.sleep(0.01)
                    continue

                # 1. MediaPipe detection on raw unflipped frame
                detection = webcam.detector.process_frame(raw_frame, draw=draw_mesh)

                # 2. Dual-hand 126-D normalization
                feature_vector = normalize_dual_hands(
                    detection.left_landmarks, detection.right_landmarks
                )

                # 3. Deep MLP direct tensor inference (0.31 ms)
                pred_result = predictor.predict(feature_vector)

                # 4. Temporal smoothing & FSM update
                smoothed_result = smoother.update(pred_result)
                state_trans = state_mgr.update(smoothed_result)

                # 5. Flip display frame for intuitive mirror view
                display_frame = cv2.flip(detection.annotated_image, 1)

                # 6. If character committed, append to sentence
                if state_trans.emitted_char:
                    self.text_builder.append_char(state_trans.emitted_char)

                # 7. Compute FPS
                fps_counter += 1
                now = time.perf_counter()
                if now - fps_timer >= 1.0:
                    current_fps = fps_counter / (now - fps_timer)
                    fps_counter = 0
                    fps_timer = now

                # 8. Encode display frame to JPEG
                _, buffer = cv2.imencode(
                    ".jpg", display_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70]
                )
                jpeg_bytes = buffer.tobytes()
                b64_frame = base64.b64encode(jpeg_bytes).decode("ascii")

                # 9. Update state thread-safely
                with self.lock:
                    self.latest_jpeg = jpeg_bytes
                    state_name = (
                        state_trans.state.value
                        if hasattr(state_trans.state, "value")
                        else str(state_trans.state)
                    )
                    self.telemetry = {
                        "fps": round(current_fps, 1),
                        "hands_detected": detection.hands_count,
                        "sign": smoothed_result.class_label,
                        "confidence": round(smoothed_result.confidence, 3),
                        "raw_sign": pred_result.class_label,
                        "top_3": pred_result.top_k,
                        "state": state_name,
                        "hold_progress": round(state_trans.hold_progress, 2),
                        "current_word": self.text_builder.current_word,
                        "sentence": self.text_builder.get_text(),
                        "image": b64_frame,
                    }

        except Exception as e:
            logger.exception("Error in vision processing loop: %s", e)
        finally:
            webcam.release()
            logger.info("Vision loop terminated cleanly.")


class RequestHandler(BaseHTTPRequestHandler):
    """
    HTTP handler serving HTML/CSS assets, MJPEG video feed, and JSON API endpoints.
    """

    service: ISLWebService = None
    html_content: str = ""

    def log_message(self, format, *args):
        # Suppress routine GET logging to keep console clean
        return

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(self.html_content.encode("utf-8"))

        elif self.path == "/video_feed":
            self.send_response(200)
            self.send_header(
                "Content-Type", "multipart/x-mixed-replace; boundary=frame"
            )
            self.end_headers()

            while self.service.running.is_set():
                frame = self.service.latest_jpeg
                if frame is not None:
                    try:
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                        self.wfile.write(frame)
                        self.wfile.write(b"\r\n")
                    except (BrokenPipeError, ConnectionResetError):
                        break
                time.sleep(0.033)  # ~30 FPS

        elif self.path == "/api/telemetry":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            with self.service.lock:
                data = json.dumps(self.service.telemetry)
            self.wfile.write(data.encode("utf-8"))

        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/api/control":
            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length)
            try:
                payload = json.loads(post_data.decode("utf-8"))
            except Exception:
                payload = {}

            with self.service.lock:
                if "action" in payload:
                    act = payload["action"]
                    if act == "space":
                        self.service.text_builder.add_space()
                    elif act == "backspace":
                        self.service.text_builder.backspace()
                    elif act == "clear":
                        self.service.text_builder.clear()

                if "paused" in payload:
                    self.service.paused = bool(payload["paused"])

                if "show_mesh" in payload:
                    self.service.show_mesh = bool(payload["show_mesh"])

                if "threshold" in payload:
                    self.service.confidence_threshold = float(payload["threshold"])

                if "hold_frames" in payload:
                    self.service.hold_frames = int(payload["hold_frames"])

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status": "ok"}')
        else:
            self.send_response(404)
            self.end_headers()


def find_free_port(start_port: int = 5000) -> int:
    """Finds an available TCP port starting from start_port."""
    for port in range(start_port, start_port + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return start_port


def run_web_app(camera_index: int = 0, port: int = 0, force_browser: bool = False):
    """
    Main launcher for the modern webview/browser interface.
    """
    if port == 0:
        port = find_free_port(5000)

    html_file = Path(__file__).resolve().parent / "web" / "index.html"
    with open(html_file, "r", encoding="utf-8") as f:
        html_code = f.read()

    service = ISLWebService(camera_index=camera_index)
    service.start_vision_pipeline()

    RequestHandler.service = service
    RequestHandler.html_content = html_code

    server = ThreadingHTTPServer(("127.0.0.1", port), RequestHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    url = f"http://127.0.0.1:{port}"
    logger.info("=================================================================")
    logger.info("  ISL RECOGNITION V2 — MODERN GLASSMORPHISM UI READY             ")
    logger.info("  Active URL: %s                                                 ", url)
    logger.info("=================================================================")

    # Attempt native desktop window via pywebview if available and not forced to browser
    use_webview = False
    if not force_browser:
        try:
            import webview

            use_webview = True
        except ImportError:
            use_webview = False

    if use_webview:
        try:
            logger.info("Launching native Desktop PyWebView window...")
            window = webview.create_window(
                title="Indian Sign Language Recognition System V2",
                url=url,
                width=1320,
                height=800,
                min_size=(1080, 680),
                text_select=False,
            )
            webview.start()
        except Exception as e:
            logger.warning("PyWebView launch failed (%s); opening in default browser instead.", e)
            webbrowser.open(url)
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
    else:
        logger.info("Opening interface in default system web browser...")
        webbrowser.open(url)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass

    # Cleanup
    logger.info("Shutting down web application...")
    service.stop_vision_pipeline()
    server.shutdown()
    logger.info("Shutdown complete.")


if __name__ == "__main__":
    run_web_app(camera_index=0)
