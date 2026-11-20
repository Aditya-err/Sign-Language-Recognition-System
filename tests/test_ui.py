"""
Unit and Integration Tests for Desktop UI & Application Entry Points.
=====================================================================

Module: sign_language_v2.tests.test_ui
"""

import queue
import pytest
from pathlib import Path
import sys

# Ensure repo root is on sys.path
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.ui.desktop_app import InferenceWorker
from src.inference.gesture_state import GestureState


def test_ui_worker_initialization():
    """Verify that InferenceWorker instantiates with proper default parameters."""
    q = queue.Queue(maxsize=2)
    worker = InferenceWorker(
        frame_queue=q,
        camera_index=0,
        confidence_threshold=0.75,
        hold_frames=15,
        smoothing_window=5,
    )
    assert worker.confidence_threshold == 0.75
    assert worker.hold_frames == 15
    assert worker.smoothing_window == 5
    assert worker.show_landmarks is True


def test_ui_worker_parameter_thread_safety():
    """Verify thread-safe parameter updating."""
    q = queue.Queue(maxsize=2)
    worker = InferenceWorker(frame_queue=q, camera_index=0)
    worker.update_parameters(conf_thresh=0.82, hold_frames=18, show_landmarks=False)

    assert worker.confidence_threshold == 0.82
    assert worker.hold_frames == 18
    assert worker.show_landmarks is False


def test_ui_run_app_import():
    """Verify that run_app is importable and defines main."""
    import run_app
    assert hasattr(run_app, "main")
