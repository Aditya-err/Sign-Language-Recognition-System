# Sign Language Recognition V2 — Project Status & Milestones

## 1. Project Locations
- **Original Project (Preserved / Untouched):** `D:\project\Ai_sign language`
- **New V2 Project:** `D:\project\sign_language_v2`

## 2. Milestone Progress Tracker

| Milestone | Status | Details |
|---|:---:|---|
| **Phase 0: Workspace Cleanup & Setup** | **COMPLETED** | Independent directory structure, Git init, `.gitignore`, backup manifest |
| **Phase 1: Core Mathematical Engine** | **COMPLETED** | 126-D dual-hand normalization, MediaPipe wrapper, collector, validator, 10 pytest tests |
| **Phase 2: Dataset, MLP Training & Eval**| **COMPLETED** | Feature extraction, stratified split, coordinate augmentation, 28k MLP, 99.61% test acc, 16 pytest tests |
| **Phase 2.1: Independent Technical Audit**| **COMPLETED** | Frame autocorrelation documented, MediaPipe handedness bug identified, real latency measured |
| **Phase 3: Real-Time Engine & OpenCV HUD**| **COMPLETED** | Handedness alignment fixed, direct tensor execution, confidence smoother, 4-state hold FSM, text builder, 26 tests passing, 29.9 FPS |
| **Phase 4: Real-World Reliability & Audit**| **COMPLETED** | 300-frame full loop benchmark (14.1 FPS on CPU, MLP 0.31ms), live recorder, independent evaluation script, 35-class testing protocol |

## 3. Environment Details
- **Active Environment:** `D:\Anaconda3\envs\ai_sign_lang\python.exe`
- **Python Version:** **Python 3.10.20**
- **Core Dependencies:**
  - `tensorflow==2.19.1`
  - `mediapipe==0.10.21`
  - `opencv-python==4.11.0.86`
  - `numpy==1.26.4`
  - `scikit-learn==1.7.2`
  - `matplotlib==3.10.9`
  - `pytest==9.2.0`
- **Host OS:** Windows (x64)
- **Local Git Repository:** Initialized on `master` branch in `D:\project\sign_language_v2`

## 4. Phase 3 Key Deliverables & Verified Metrics
- **Real-Time Inference Engine:** `src/inference/predictor.py` (direct tensor execution `model(x)` in 0.15 ms).
- **Webcam Manager with Handedness Alignment:** `src/inference/webcam.py` (unflipped frame to MediaPipe, flipped frame for display).
- **Prediction Smoother:** `src/inference/smoothing.py` (rolling window of 7 frames, threshold 0.70).
- **Gesture State Machine:** `src/inference/gesture_state.py` (4-state FSM: IDLE $\to$ HOLDING $\to$ COMMITTED $\to$ LOCKED).
- **Text Builder:** `src/inference/text_builder.py` (interactive character appending, spaces, backspaces, clear).
- **OpenCV HUD Application:** `src/ui/opencv_app.py` (live HUD with hold progress bar, Top-3 hypotheses, text display).
- **Hardware Latency Benchmark:** `src/inference/benchmark.py` (real hardware profiling: 33.41 ms total latency, 29.9 FPS).
- **Automated Tests:** 26 automated unit and pipeline tests passing under pytest.
