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
| **Phase 3: Real-Time Engine & UI** | **READY** | Real-time inference engine, prediction smoothing, temporal state machine, OpenCV HUD |

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

## 4. Phase 2 Key Deliverables & Verified Metrics
- **Extracted Dataset:** 5,100 valid samples across 35 ISL classes (digits 1-9, letters A-Z).
- **Extraction Success Rate:** 97.14% (150 failed/cropped images rejected cleanly).
- **Splits:** 3,570 Train (70%), 765 Validation (15%), 765 Test (15%).
- **Model:** Keras Sequential MLP (28,515 parameters, 111 KB).
- **Training Duration:** 12.83 seconds on CPU (EarlyStopping at epoch 27, best weights restored from epoch 15).
- **Test Performance (Sealed Partition):**
  - **Top-1 Accuracy:** 99.61% (762 / 765)
  - **Top-3 Accuracy:** 99.87% (764 / 765)
  - **Macro Precision / Recall / F1:** 0.9960
  - **Weighted F1:** 0.9961
- **Testing:** 16 automated tests passing in 10.63 seconds.
- **Reports & Visualizations:**
  - `reports/evaluation_report.md`
  - `reports/dataset_analysis.md`
  - `reports/figures/confusion_matrix.png`
  - `reports/figures/training_history.png`
  - `reports/figures/class_distribution.png`
  - `reports/figures/hand_types.png`

## 5. Next Phase Roadmap (Phase 3)
- Real-time webcam inference engine with background threading.
- Prediction smoothing (rolling window majority vote / exponential moving average).
- Temporal State Machine for fingerspelling word assembly (holding sign for $T$ frames commits letter).
- Text-to-Speech (TTS) integration using Python `pyttsx3`.
- Interactive OpenCV HUD with landmark skeleton overlay, confidence bar, and sentence buffer.
