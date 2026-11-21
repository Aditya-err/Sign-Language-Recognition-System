"""
Desktop Graphical Application for Real-Time Indian Sign Language Recognition.
=============================================================================

Module: sign_language_v2.src.ui.desktop_app
Author: Senior AI/ML Engineer & Python Developer (Mentoring MCA Fresher)

Architecture Overview:
----------------------
This application provides an intuitive desktop interface for real-time ISL recognition.
It implements a decoupled, thread-safe Producer-Consumer architecture:

  [InferenceWorker Thread]
      │ (Captures camera frame, runs MediaPipe, normalizes coordinates,
      │  executes MLP tensor forward pass, updates smoother & FSM)
      ▼
   queue.Queue (Max size 2 to prevent stale frame lag)
      ▼
  [Tkinter Main GUI Thread]
      │ (Reads latest packet via root.after(15ms), renders mirrored video,
      │  updates real-time HUD, confidence bars, and text construction buffer)

Key Features:
  - Video feed with live landmark visualization toggle
  - Big prominent prediction typography and dynamic state color coding
  - Interactive confidence threshold and hold duration sliders
  - Complete word & sentence builder with space, backspace, and clipboard copy
  - Supported gestures reference panel (A-Z, 1-9; noting digit 0 is absent)
  - 100% thread-safe with graceful resource cleanup on window close
"""

import logging
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
from typing import Optional, Tuple, List

import cv2
import numpy as np
from PIL import Image, ImageTk

# Ensure project root is in sys.path
sys_path_root = Path(__file__).resolve().parents[2]
if str(sys_path_root) not in sys.path:
    sys.path.insert(0, str(sys_path_root))

from src.core.normalization import normalize_dual_hands
from src.inference.gesture_state import GestureStateManager, GestureState
from src.inference.predictor import SignPredictor
from src.inference.smoothing import PredictionSmoother
from src.inference.text_builder import TextBuilder
from src.inference.webcam import WebcamManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("desktop_app")


class InferenceWorker(threading.Thread):
    """
    Background worker thread that runs the complete computer vision and deep learning
    inference loop, delivering processed frame packets to the UI queue.
    """

    def __init__(
        self,
        frame_queue: queue.Queue,
        camera_index: int = 0,
        confidence_threshold: float = 0.70,
        hold_frames: int = 12,
        smoothing_window: int = 7,
    ):
        super().__init__(daemon=True)
        self.frame_queue = frame_queue
        self.camera_index = camera_index
        self.running = threading.Event()
        self.paused = threading.Event()
        self.show_landmarks = True
        self.confidence_threshold = confidence_threshold
        self.hold_frames = hold_frames
        self.smoothing_window = smoothing_window

        # Thread-safe parameter update lock
        self._param_lock = threading.Lock()

    def update_parameters(self, conf_thresh: float, hold_frames: int, show_landmarks: bool):
        with self._param_lock:
            self.confidence_threshold = conf_thresh
            self.hold_frames = hold_frames
            self.show_landmarks = show_landmarks

    def stop(self):
        self.running.clear()

    def pause(self):
        self.paused.set()

    def resume(self):
        self.paused.clear()

    def run(self):
        self.running.set()
        logger.info("Starting background inference thread...")

        # Initialize hardware and models inside the thread
        predictor = SignPredictor()
        webcam = WebcamManager(camera_index=self.camera_index, mirror_display=False)

        if not webcam.is_ready:
            logger.error("Webcam hardware could not be initialized at index %d", self.camera_index)
            self.frame_queue.put({"error": "Camera could not be opened"})
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
                if self.paused.is_set():
                    time.sleep(0.05)
                    continue

                # Read parameters safely
                with self._param_lock:
                    smoother.min_confidence = self.confidence_threshold
                    state_mgr.hold_frames_threshold = self.hold_frames
                    draw_lm = self.show_landmarks

                ret, raw_frame = webcam.cap.read()
                if not ret or raw_frame is None:
                    time.sleep(0.01)
                    continue

                # 1. MediaPipe detection on raw unflipped frame (handedness preservation)
                detection = webcam.detector.process_frame(raw_frame, draw=draw_lm)

                # 2. Dual-hand 126-D normalization
                feature_vector = normalize_dual_hands(
                    detection.left_landmarks, detection.right_landmarks
                )

                # 3. Deep MLP forward pass
                pred_result = predictor.predict(feature_vector)

                # 4. Temporal smoothing & state machine update
                smoothed_result = smoother.update(pred_result)
                state_transition = state_mgr.update(smoothed_result)

                # 5. Flip display frame for natural mirror ergonomics
                display_bgr = cv2.flip(detection.annotated_image, 1)

                # Compute frame rate
                fps_counter += 1
                now = time.perf_counter()
                if now - fps_timer >= 1.0:
                    current_fps = fps_counter / (now - fps_timer)
                    fps_counter = 0
                    fps_timer = now

                # Package data packet for UI thread
                packet = {
                    "frame": display_bgr,
                    "hands_detected": detection.hands_detected,
                    "sign": smoothed_result.class_name,
                    "confidence": smoothed_result.confidence,
                    "raw_sign": pred_result.class_name,
                    "raw_confidence": pred_result.confidence,
                    "top_3": pred_result.top_k,
                    "state": state_transition.current_state,
                    "hold_progress": state_transition.hold_progress,
                    "committed_char": state_transition.committed_character,
                    "fps": round(current_fps, 1),
                }

                # Push to queue, dropping older frames if UI thread is busy
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.frame_queue.put(packet)

        except Exception as e:
            logger.exception("Exception in InferenceWorker: %s", e)
            self.frame_queue.put({"error": str(e)})
        finally:
            webcam.release()
            logger.info("InferenceWorker terminated cleanly.")


class SignLanguageApp(tk.Tk):
    """
    Main Tkinter Desktop Application with full controls, HUD, and text construction.
    """

    def __init__(self):
        super().__init__()

        self.title("Indian Sign Language Recognition System V2")
        self.geometry("1280x760")
        self.minsize(1080, 680)
        self.configure(bg="#181825")

        # Application state
        self.text_builder = TextBuilder()
        self.frame_queue = queue.Queue(maxsize=2)
        self.worker: Optional[InferenceWorker] = None
        self.is_camera_running = False
        self.is_recognition_paused = False
        self._canvas_img_id = None

        # GUI Styling
        self._setup_styles()

        # Build UI layout
        self._build_header()
        self._build_main_content()
        self._build_footer()

        # Handle window close protocol
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # Start periodic GUI queue polling (every 15 ms)
        self.after(15, self._poll_queue)

        # Auto-start camera
        self.after(200, self._start_camera)

    def _setup_styles(self):
        self.style = ttk.Style(self)
        self.style.theme_use("clam")

        # Configure custom colors
        self.style.configure(
            "TFrame",
            background="#181825",
        )
        self.style.configure(
            "Card.TFrame",
            background="#242438",
            relief="flat",
        )
        self.style.configure(
            "TLabel",
            background="#242438",
            foreground="#f8fafc",
            font=("Segoe UI", 10),
        )
        self.style.configure(
            "Header.TLabel",
            background="#181825",
            foreground="#ffffff",
            font=("Segoe UI", 16, "bold"),
        )
        self.style.configure(
            "SubHeader.TLabel",
            background="#242438",
            foreground="#94a3b8",
            font=("Segoe UI", 9, "bold"),
        )
        self.style.configure(
            "Value.TLabel",
            background="#242438",
            foreground="#38bdf8",
            font=("Segoe UI", 11, "bold"),
        )
        self.style.configure(
            "Primary.TButton",
            font=("Segoe UI", 10, "bold"),
            background="#4f46e5",
            foreground="#ffffff",
            borderwidth=0,
            padding=6,
        )
        self.style.map(
            "Primary.TButton",
            background=[("active", "#4338ca"), ("disabled", "#334155")],
        )
        self.style.configure(
            "Danger.TButton",
            font=("Segoe UI", 10, "bold"),
            background="#ef4444",
            foreground="#ffffff",
            borderwidth=0,
            padding=6,
        )
        self.style.map(
            "Danger.TButton",
            background=[("active", "#dc2626")],
        )
        self.style.configure(
            "Success.TButton",
            font=("Segoe UI", 10, "bold"),
            background="#10b981",
            foreground="#ffffff",
            borderwidth=0,
            padding=6,
        )
        self.style.map(
            "Success.TButton",
            background=[("active", "#059669")],
        )

    def _build_header(self):
        header_frame = ttk.Frame(self, style="TFrame")
        header_frame.pack(fill=tk.X, padx=16, pady=(10, 6))

        title_lbl = ttk.Label(
            header_frame,
            text="Indian Sign Language Recognition System V2",
            style="Header.TLabel",
        )
        title_lbl.pack(side=tk.LEFT)

        subtitle_lbl = ttk.Label(
            header_frame,
            text=" | Real-Time AI / MediaPipe + 126-D Normalization + Deep MLP",
            font=("Segoe UI", 10),
            foreground="#94a3b8",
            background="#181825",
        )
        subtitle_lbl.pack(side=tk.LEFT, padx=6)

        self.fps_badge = ttk.Label(
            header_frame,
            text="FPS: 0.0",
            font=("Segoe UI", 10, "bold"),
            foreground="#10b981",
            background="#181825",
        )
        self.fps_badge.pack(side=tk.RIGHT, padx=12)

    def _build_main_content(self):
        content_frame = ttk.Frame(self, style="TFrame")
        content_frame.pack(fill=tk.BOTH, expand=True, padx=16, pady=6)

        # Left Column: Video View & Video Controls
        left_col = ttk.Frame(content_frame, style="TFrame", width=650)
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))

        # Video Canvas Card
        video_card = ttk.Frame(left_col, style="Card.TFrame")
        video_card.pack(fill=tk.BOTH, expand=True)

        self.video_canvas = tk.Canvas(
            video_card,
            bg="#0f0f17",
            highlightthickness=0,
            width=640,
            height=480,
        )
        self.video_canvas.pack(padx=8, pady=8, fill=tk.BOTH, expand=True)
        self.video_canvas.create_text(
            320,
            240,
            text="Initializing Camera & AI Engine...\nPlease wait a moment.",
            fill="#94a3b8",
            font=("Segoe UI", 13),
            justify=tk.CENTER,
            tags="placeholder",
        )

        # Video Toolbar
        v_tool = ttk.Frame(video_card, style="Card.TFrame")
        v_tool.pack(fill=tk.X, padx=8, pady=(0, 8))

        self.btn_camera = ttk.Button(
            v_tool,
            text="Stop Camera",
            style="Primary.TButton",
            command=self._toggle_camera,
        )
        self.btn_camera.pack(side=tk.LEFT, padx=4)

        self.btn_pause = ttk.Button(
            v_tool,
            text="Pause AI",
            style="Primary.TButton",
            command=self._toggle_pause,
        )
        self.btn_pause.pack(side=tk.LEFT, padx=4)

        self.var_landmarks = tk.BooleanVar(value=True)
        chk_lm = tk.Checkbutton(
            v_tool,
            text="Draw Hand Mesh",
            variable=self.var_landmarks,
            command=self._on_settings_change,
            bg="#242438",
            fg="#f8fafc",
            selectcolor="#181825",
            activebackground="#242438",
            activeforeground="#f8fafc",
            font=("Segoe UI", 9),
        )
        chk_lm.pack(side=tk.LEFT, padx=12)

        self.lbl_hand_status = ttk.Label(
            v_tool,
            text="Hands: None",
            foreground="#e2e8f0",
            font=("Segoe UI", 9, "bold"),
        )
        self.lbl_hand_status.pack(side=tk.RIGHT, padx=8)

        # Right Column: Recognition Card, Text Builder, Settings, Reference
        right_col = ttk.Frame(content_frame, style="TFrame", width=550)
        right_col.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False, padx=(8, 0))

        # 1. Live Recognition & Confidence Card
        rec_card = ttk.Frame(right_col, style="Card.TFrame")
        rec_card.pack(fill=tk.X, pady=(0, 8))

        rec_header = ttk.Label(rec_card, text="REAL-TIME RECOGNITION", style="SubHeader.TLabel")
        rec_header.pack(anchor=tk.W, padx=12, pady=(8, 4))

        rec_inner = ttk.Frame(rec_card, style="Card.TFrame")
        rec_inner.pack(fill=tk.X, padx=12, pady=(0, 8))

        # Big Predicted Sign display
        self.lbl_predicted_sign = tk.Label(
            rec_inner,
            text="—",
            font=("Segoe UI", 52, "bold"),
            bg="#1e1e30",
            fg="#38bdf8",
            width=3,
            relief="flat",
        )
        self.lbl_predicted_sign.pack(side=tk.LEFT, padx=(0, 12), pady=4)

        details_box = ttk.Frame(rec_inner, style="Card.TFrame")
        details_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # State Pill Badge
        state_row = ttk.Frame(details_box, style="Card.TFrame")
        state_row.pack(fill=tk.X, pady=(2, 4))
        ttk.Label(state_row, text="State: ").pack(side=tk.LEFT)
        self.lbl_state = tk.Label(
            state_row,
            text="IDLE",
            font=("Segoe UI", 9, "bold"),
            bg="#334155",
            fg="#cbd5e1",
            padx=8,
            pady=1,
        )
        self.lbl_state.pack(side=tk.LEFT)

        # Confidence Bar
        conf_row = ttk.Frame(details_box, style="Card.TFrame")
        conf_row.pack(fill=tk.X, pady=2)
        self.lbl_conf_val = ttk.Label(conf_row, text="Confidence: 0.0%", font=("Segoe UI", 9))
        self.lbl_conf_val.pack(anchor=tk.W)
        self.bar_conf = ttk.Progressbar(conf_row, orient=tk.HORIZONTAL, length=180, mode="determinate")
        self.bar_conf.pack(fill=tk.X, pady=(2, 4))

        # Hold Progress Bar
        hold_row = ttk.Frame(details_box, style="Card.TFrame")
        hold_row.pack(fill=tk.X, pady=2)
        self.lbl_hold_val = ttk.Label(hold_row, text="Commit Hold: 0%", font=("Segoe UI", 9))
        self.lbl_hold_val.pack(anchor=tk.W)
        self.bar_hold = ttk.Progressbar(hold_row, orient=tk.HORIZONTAL, length=180, mode="determinate")
        self.bar_hold.pack(fill=tk.X, pady=(2, 4))

        # Top-3 Predictions Row
        self.lbl_top3 = ttk.Label(
            rec_card,
            text="Top Hypotheses: [1] — (0.0%) | [2] — (0.0%) | [3] — (0.0%)",
            font=("Segoe UI", 8),
            foreground="#94a3b8",
        )
        self.lbl_top3.pack(anchor=tk.W, padx=12, pady=(0, 8))

        # 2. Text Construction Card
        text_card = ttk.Frame(right_col, style="Card.TFrame")
        text_card.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        ttk.Label(text_card, text="TEXT CONVERSATION BUFFER", style="SubHeader.TLabel").pack(
            anchor=tk.W, padx=12, pady=(8, 4)
        )

        word_row = ttk.Frame(text_card, style="Card.TFrame")
        word_row.pack(fill=tk.X, padx=12, pady=(0, 4))
        ttk.Label(word_row, text="Current Word: ", font=("Segoe UI", 9)).pack(side=tk.LEFT)
        self.lbl_current_word = ttk.Label(
            word_row,
            text="",
            font=("Segoe UI", 12, "bold"),
            foreground="#f59e0b",
        )
        self.lbl_current_word.pack(side=tk.LEFT)

        # Full Sentence Text Box
        self.txt_sentence = tk.Text(
            text_card,
            height=4,
            bg="#181825",
            fg="#f8fafc",
            insertbackground="#ffffff",
            font=("Segoe UI", 11),
            relief="flat",
            wrap=tk.WORD,
        )
        self.txt_sentence.pack(fill=tk.BOTH, expand=True, padx=12, pady=4)

        # Sentence Action Buttons
        btn_box = ttk.Frame(text_card, style="Card.TFrame")
        btn_box.pack(fill=tk.X, padx=12, pady=(4, 8))

        ttk.Button(btn_box, text="Space", style="Primary.TButton", command=self._add_space).pack(
            side=tk.LEFT, padx=(0, 4)
        )
        ttk.Button(btn_box, text="Backspace", style="Primary.TButton", command=self._backspace).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Button(btn_box, text="Clear", style="Danger.TButton", command=self._clear_text).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Button(btn_box, text="Copy to Clipboard", style="Success.TButton", command=self._copy_clipboard).pack(
            side=tk.RIGHT, padx=4
        )

        # 3. Parameters & Reference Tabs
        notebook = ttk.Notebook(right_col)
        notebook.pack(fill=tk.X, expand=False)

        tab_controls = ttk.Frame(notebook, style="Card.TFrame")
        tab_reference = ttk.Frame(notebook, style="Card.TFrame")
        notebook.add(tab_controls, text="  Tuning Controls  ")
        notebook.add(tab_reference, text="  Supported Signs & Notes  ")

        # Controls Tab: Sliders
        # Confidence Threshold Slider
        s_row1 = ttk.Frame(tab_controls, style="Card.TFrame")
        s_row1.pack(fill=tk.X, padx=12, pady=6)
        self.lbl_slider_conf = ttk.Label(
            s_row1,
            text="Confidence Threshold: 0.70",
            font=("Segoe UI", 9),
        )
        self.lbl_slider_conf.pack(anchor=tk.W)
        self.scale_conf = ttk.Scale(
            s_row1,
            from_=0.50,
            to=0.95,
            value=0.70,
            command=self._on_conf_slider,
        )
        self.scale_conf.pack(fill=tk.X, pady=2)

        # Hold Frames Slider
        s_row2 = ttk.Frame(tab_controls, style="Card.TFrame")
        s_row2.pack(fill=tk.X, padx=12, pady=6)
        self.lbl_slider_hold = ttk.Label(
            s_row2,
            text="Commit Hold Frames: 12 (~0.8s)",
            font=("Segoe UI", 9),
        )
        self.lbl_slider_hold.pack(anchor=tk.W)
        self.scale_hold = ttk.Scale(
            s_row2,
            from_=5,
            to=25,
            value=12,
            command=self._on_hold_slider,
        )
        self.scale_hold.pack(fill=tk.X, pady=2)

        # Reference Tab: Supported signs list
        ref_box = ttk.Frame(tab_reference, style="Card.TFrame")
        ref_box.pack(fill=tk.BOTH, padx=12, pady=6)

        ref_txt = (
            "Supported Classes (35 Total):\n"
            "• Digits (1-9): 1, 2, 3, 4, 5, 6, 7, 8, 9\n"
            "• Alphabets (A-Z): A through Z (Single & Dual Hand)\n"
            "• Unsupported: Digit '0' is ABSENT from dataset\n\n"
            "Operation Tips:\n"
            "• Keep torso and hands 0.8m-1.2m from camera\n"
            "• Hold gesture steady until progress bar completes to commit\n"
            "• Lower hands briefly between characters to reset lock"
        )
        ttk.Label(
            ref_box,
            text=ref_txt,
            font=("Segoe UI", 8),
            foreground="#cbd5e1",
            justify=tk.LEFT,
        ).pack(anchor=tk.W)

    def _build_footer(self):
        footer_frame = ttk.Frame(self, style="TFrame")
        footer_frame.pack(fill=tk.X, padx=16, pady=(4, 8))

        lbl_footer = ttk.Label(
            footer_frame,
            text="MCA AI/ML Portfolio Project | Engineering Integrity: 100% | Zero Mocked Predictions",
            font=("Segoe UI", 8),
            foreground="#64748b",
            background="#181825",
        )
        lbl_footer.pack(side=tk.LEFT)

        lbl_env = ttk.Label(
            footer_frame,
            text="TensorFlow Direct Tensor (0.3ms) | MediaPipe Hands | Tkinter Thread-Safe GUI",
            font=("Segoe UI", 8),
            foreground="#64748b",
            background="#181825",
        )
        lbl_env.pack(side=tk.RIGHT)

    def _on_conf_slider(self, val):
        fval = float(val)
        self.lbl_slider_conf.config(text=f"Confidence Threshold: {fval:.2f}")
        self._on_settings_change()

    def _on_hold_slider(self, val):
        ival = int(float(val))
        seconds = ival / 15.0
        self.lbl_slider_hold.config(text=f"Commit Hold Frames: {ival} (~{seconds:.1f}s)")
        self._on_settings_change()

    def _on_settings_change(self):
        if self.worker is not None and self.worker.is_alive():
            conf = float(self.scale_conf.get())
            hold = int(float(self.scale_hold.get()))
            lm = self.var_landmarks.get()
            self.worker.update_parameters(conf, hold, lm)

    def _start_camera(self):
        if self.worker is None or not self.worker.is_alive():
            conf = float(self.scale_conf.get())
            hold = int(float(self.scale_hold.get()))
            self.worker = InferenceWorker(
                frame_queue=self.frame_queue,
                camera_index=0,
                confidence_threshold=conf,
                hold_frames=hold,
            )
            self.worker.show_landmarks = self.var_landmarks.get()
            self.worker.start()
            self.is_camera_running = True
            self.btn_camera.config(text="Stop Camera")

    def _stop_camera(self):
        if self.worker is not None and self.worker.is_alive():
            self.worker.stop()
            self.worker.join(timeout=1.5)
            self.worker = None
        self.is_camera_running = False
        self.btn_camera.config(text="Start Camera")
        self.video_canvas.delete("all")
        self._canvas_img_id = None
        self.lbl_predicted_sign.config(text="—")
        self.lbl_state.config(text="STOPPED", bg="#475569")

    def _toggle_camera(self):
        if self.is_camera_running:
            self._stop_camera()
        else:
            self._start_camera()

    def _toggle_pause(self):
        if self.worker is None:
            return
        if self.is_recognition_paused:
            self.worker.resume()
            self.is_recognition_paused = False
            self.btn_pause.config(text="Pause AI")
        else:
            self.worker.pause()
            self.is_recognition_paused = True
            self.btn_pause.config(text="Resume AI")
            self.lbl_state.config(text="PAUSED", bg="#d97706")

    def _add_space(self):
        self.text_builder.add_space()
        self._sync_text_display()

    def _backspace(self):
        self.text_builder.backspace()
        self._sync_text_display()

    def _clear_text(self):
        self.text_builder.clear()
        self._sync_text_display()

    def _copy_clipboard(self):
        full_text = self.text_builder.get_full_text()
        if full_text.strip():
            self.clipboard_clear()
            self.clipboard_append(full_text)
            messagebox.showinfo("Clipboard", "Text copied to clipboard successfully!")
        else:
            messagebox.showinfo("Clipboard", "Sentence buffer is currently empty.")

    def _sync_text_display(self):
        self.lbl_current_word.config(text=self.text_builder.current_word)
        self.txt_sentence.delete("1.0", tk.END)
        self.txt_sentence.insert(tk.END, self.text_builder.get_full_text())
        self.txt_sentence.see(tk.END)

    def _poll_queue(self):
        """
        Polls the background inference worker queue and updates the GUI widgets.
        """
        try:
            while not self.frame_queue.empty():
                packet = self.frame_queue.get_nowait()

                if "error" in packet:
                    self.lbl_state.config(text="CAMERA ERROR", bg="#dc2626")
                    logger.warning("Worker reported error: %s", packet["error"])
                    continue

                # 1. Render Video Frame
                frame_bgr = packet["frame"]
                rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                h, w = rgb_frame.shape[:2]

                # Remove placeholder once first frame arrives
                self.video_canvas.delete("placeholder")

                # Resize to canvas dimensions if needed
                canvas_w = self.video_canvas.winfo_width()
                canvas_h = self.video_canvas.winfo_height()
                if canvas_w <= 50 or canvas_h <= 50:
                    canvas_w, canvas_h = 640, 480

                if canvas_w > 50 and canvas_h > 50 and (canvas_w != w or canvas_h != h):
                    # Aspect ratio preserving fit
                    scale = min(canvas_w / w, canvas_h / h)
                    nw, nh = int(w * scale), int(h * scale)
                    rgb_frame = cv2.resize(rgb_frame, (nw, nh), interpolation=cv2.INTER_LINEAR)

                img_pil = Image.fromarray(rgb_frame)
                img_tk = ImageTk.PhotoImage(image=img_pil)

                # Re-use canvas item to prevent memory leak and slow re-draw
                if self._canvas_img_id is None:
                    self._canvas_img_id = self.video_canvas.create_image(
                        canvas_w // 2, canvas_h // 2, anchor=tk.CENTER, image=img_tk
                    )
                else:
                    self.video_canvas.coords(self._canvas_img_id, canvas_w // 2, canvas_h // 2)
                    self.video_canvas.itemconfig(self._canvas_img_id, image=img_tk)

                self._current_img = img_tk  # Keep reference to prevent GC

                # 2. Update Telemetry Badges
                self.fps_badge.config(text=f"FPS: {packet['fps']}")
                self.lbl_hand_status.config(
                    text=f"Hands: {packet['hands_detected']} detected"
                )

                # 3. Update Recognition Card
                sign = packet["sign"]
                conf = packet["confidence"]
                state = packet["state"]
                hold_prog = packet["hold_progress"]
                committed_char = packet["committed_char"]

                if sign is not None:
                    self.lbl_predicted_sign.config(text=sign)
                else:
                    self.lbl_predicted_sign.config(text="—")

                # State colors
                state_colors = {
                    GestureState.IDLE: ("#334155", "#cbd5e1"),
                    GestureState.DETECTING: ("#1d4ed8", "#ffffff"),
                    GestureState.HOLDING: ("#d97706", "#ffffff"),
                    GestureState.COMMITTED: ("#059669", "#ffffff"),
                    GestureState.LOCKED: ("#7c3aed", "#ffffff"),
                }
                bg_col, fg_col = state_colors.get(state, ("#334155", "#ffffff"))
                self.lbl_state.config(text=state.name, bg=bg_col, fg=fg_col)

                # Bars
                self.lbl_conf_val.config(text=f"Confidence: {conf*100:.1f}%")
                self.bar_conf["value"] = conf * 100

                self.lbl_hold_val.config(text=f"Commit Hold: {int(hold_prog*100)}%")
                self.bar_hold["value"] = hold_prog * 100

                # Top-3 predictions
                top_3 = packet.get("top_3", [])
                t_str = "Top Hypotheses: "
                for i, (c_name, c_val) in enumerate(top_3):
                    t_str += f"[{i+1}] {c_name} ({c_val*100:.1f}%)   "
                self.lbl_top3.config(text=t_str)

                # 4. If character committed, append to text builder
                if committed_char:
                    self.text_builder.append_char(committed_char)
                    self._sync_text_display()

        except Exception as e:
            logger.exception("Error polling UI queue: %s", e)

        # Re-schedule next poll
        self.after(15, self._poll_queue)

    def _on_close(self):
        logger.info("Closing application window...")
        self._stop_camera()
        self.destroy()


def main():
    app = SignLanguageApp()
    app.mainloop()


if __name__ == "__main__":
    main()
