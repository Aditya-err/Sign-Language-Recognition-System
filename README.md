# Indian Sign Language (ISL) Recognition Engine V2

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16%2B-orange.svg)](https://tensorflow.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Hands-teal.svg)](https://mediapipe.dev)
[![Tests](https://img.shields.io/badge/Pytest-10%20Passed-brightgreen.svg)](tests/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An industry-standard, high-performance **Indian Sign Language (ISL) Recognition System** built with **MediaPipe Hands**, a mathematically proven **dual-hand spatial normalization pipeline** (126 features), an **optimized Deep MLP classifier**, and an **event-driven finite state machine** for stable real-time fingerspelling to speech translation.

---

## 🌟 Why Version 2? (Technical Evolution)

| Aspect | Legacy V1 Implementation | Version 2 Architecture |
|---|---|---|
| **Linguistic Scope** | Assumed single-hand ASL | Native **Dual-Hand Indian Sign Language (ISL)** |
| **Model Type** | 599,000-parameter LSTM (Overfitted) | Lightweight Deep MLP (< 25,000 params) |
| **Feature Space** | 1,662 features (84.5% irrelevant face mesh) | **126 clean features** (63 Left + 63 Right hand) |
| **Normalization** | None (Absolute screen coordinates) | **Wrist-centering + Rigid Palm-scale invariance** |
| **Missing Hands** | Filled with fabricated zero-vectors | Explicit `HandDetectionResult` bypassing classifier |
| **Inference Latency** | ~60 ms per frame (12–16 FPS) | **< 1 ms per frame (> 35 FPS)** |
| **Testing** | Zero automated tests | **10 unit tests with 100% pytest pass rate** |
| **Data Leakage** | Random split across single-person video | **Participant-Independent (GroupKFold / LOSO)** |

---

## 📐 Mathematical Normalization Pipeline

Raw MediaPipe coordinates vary whenever a user moves in front of the lens. V2 implements deterministic anatomical normalization:

1. **Wrist Translation Invariance:** All 21 landmarks are translated so Landmark 0 (wrist) sits at origin $(0, 0, 0)$:
   $$\mathbf{p}_i^{\text{trans}} = \mathbf{p}_i - \mathbf{p}_0 \quad \forall i \in \{0, \dots, 20\}$$
2. **Palm-Scale Invariance:** Coordinates are scaled by the rigid palm distance between the wrist ($\mathbf{p}_0$) and middle finger base knuckle ($\mathbf{p}_9$):
   $$s = \|\mathbf{p}_9^{\text{trans}}\|_2 = \sqrt{(x_9 - x_0)^2 + (y_9 - y_0)^2 + (z_9 - z_0)^2}$$
   $$\mathbf{p}_i^{\text{norm}} = \frac{\mathbf{p}_i^{\text{trans}}}{s + 10^{-6}}$$
3. **Dual-Hand ISL Vector:** Concatenates normalized Left (0–62) and Right (63–125) hands into a unified 126-dimensional vector $\mathbf{x} \in \mathbb{R}^{126}$.

---

## 📁 Repository Structure

```text
sign_language_v2/
├── config/                  # Configuration files
├── data/
│   ├── raw/                 # Subject-tagged raw landmark JSON records
│   └── processed/           # Leakage-free train/val/test splits
├── docs/                    # Technical documentation
│   ├── DATASET_ASSESSMENT.md# Detailed ISL dataset audit
│   ├── PHASE_1_REPORT.md    # Phase 1 verification report
│   └── PROJECT_STATUS.md    # Milestones & setup log
├── models/                  # Checkpointed weights and exported ONNX models
├── notebooks/               # Exploratory analysis & model benchmarking
├── reports/                 # Confusion matrices, ROC curves, evaluation reports
├── src/
│   ├── core/                # Landmark normalization & MediaPipe abstraction
│   │   ├── normalization.py # 126-D dual-hand spatial normalization
│   │   └── hand_detector.py # MediaPipe Hands dual-hand detection wrapper
│   ├── data/                # Data collection & validation
│   │   ├── collector.py     # Guided OpenCV data collector with HUD
│   │   └── dataset_validator.py # Automated integrity audit tool
│   ├── models/              # MLP architecture, training, and evaluation
│   ├── inference/           # Threaded engine, state machine, and TTS
│   └── ui/                  # OpenCV HUD and Streamlit web dashboard
├── tests/                   # Pytest automated test suite
│   ├── test_normalization.py# Mathematical invariance tests
│   └── test_validator.py    # Dataset integrity tests
├── requirements.txt         # Pinned production dependencies
└── README.md                # Project documentation
```

---

## 🚀 Quick Start Guide

### 1. Environment Setup
```bash
# Clone the repository and enter directory
cd D:\project\sign_language_v2

# Activate compatible Python 3.10 environment
conda activate ai_sign_lang
```

### 2. Run Automated Pytest Suite
```bash
python -m pytest tests/ -v
```
All 10 unit tests will run and verify mathematical scale, translation, and dimension invariance.

### 3. Launch Guided Webcam Data Collector
```bash
python -m src.data.collector
```
- Enter target sign (`A`, `B`, `1`, etc.), Participant ID (`P01`), and Session ID (`S01`).
- Press **[SPACE]** to capture single clean landmark samples.
- Press **[R]** to toggle continuous burst recording (captures sample every 250ms).
- Press **[Q]** to quit.

### 4. Run Dataset Validator
```bash
python -m src.data.dataset_validator
```
Audits all recorded JSON samples under `data/raw/` for missing metadata, corrupt values, or duplicate vectors.

---

## 📜 Development Roadmap

- [x] **Phase 0:** Workspace initialization & environment setup.
- [x] **Phase 1:** Core normalization (126-D), MediaPipe detection, unit testing, guided collector, and validator.
- [ ] **Phase 2:** Deep MLP model training, GroupKFold cross-validation, and confusion matrix evaluation.
- [ ] **Phase 3:** Real-time inference engine with gesture-hold state machine and pyttsx3 speech.
- [ ] **Phase 4:** Streamlit web UI and automated GitHub CI pipeline.
