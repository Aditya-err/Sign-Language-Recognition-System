# Indian Sign Language (ISL) Recognition Engine V2

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19-orange.svg)](https://tensorflow.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Hands-teal.svg)](https://mediapipe.dev)
[![Tests](https://img.shields.io/badge/Pytest-26%20Passed-brightgreen.svg)](tests/)
[![Real--Time](https://img.shields.io/badge/Real--Time-29.9%20FPS-success.svg)](src/ui/opencv_app.py)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An industry-standard, high-performance **Indian Sign Language (ISL) Recognition System** built with **MediaPipe Hands**, a mathematically proven **dual-hand spatial normalization pipeline** (126 features), an **optimized Deep MLP classifier** (28,515 parameters), and a **temporal stabilization state machine** for fluid real-time fingerspelling to text translation.

---

## 🌟 Why Version 2? (Technical Evolution)

| Aspect | Legacy V1 Implementation | Version 2 Architecture |
|---|---|---|
| **Linguistic Scope** | Assumed single-hand ASL | Native **Dual-Hand Indian Sign Language (ISL)** |
| **Model Type** | 599,000-parameter LSTM (Overfitted) | **Lightweight Deep MLP (28,515 params, 111 KB)** |
| **Feature Space** | 1,662 features (84.5% irrelevant face mesh) | **126 clean features** (63 Left + 63 Right hand) |
| **Normalization** | None (Absolute screen coordinates) | **Wrist-centering + Rigid Palm-scale invariance** |
| **Missing Hands** | Filled with fabricated zero-vectors | Explicit `has_hands=False` check bypassing classifier |
| **Inference Latency** | ~60 ms per frame (12–16 FPS) | **0.15 ms forward pass (29.9 FPS end-to-end)** |
| **Real-Time UI** | Mock predictions & synthetic charts | **Live OpenCV HUD with temporal hold state machine** |
| **Duplicate Prevention**| None ("AAAAA" spam on sign hold) | **Single-emission FSM with cooldown & progress bar** |
| **Automated Tests** | Zero automated tests | **26 unit/integration tests with 100% pytest pass rate** |

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

## ⚡ Real-Time Hardware Performance

Measured on standard CPU hardware across 40 live frames:

- **Webcam Frame Grab (`cap.read`):** $7.03$ ms
- **MediaPipe Hands Tracking:** $26.18$ ms
- **Spatial Normalization:** $0.025$ ms
- **Deep MLP Forward Pass (`model(x)`):** **$0.15$ ms**
- **Temporal Smoother & State Machine:** $0.017$ ms
- **Total Frame Latency:** **$33.41$ ms ($29.9$ FPS)**

---

## 📁 Repository Structure

```text
sign_language_v2/
├── config/                  # Configuration files
├── data/
│   ├── raw/                 # Subject-tagged raw landmark JSON records
│   └── processed/           # Processed features & stratified splits (git-ignored)
│       ├── extraction_metadata.json
│       ├── label_encoder.json
│       └── split_info.json
├── docs/                    # Technical documentation
│   ├── DATASET_ASSESSMENT.md# Detailed ISL dataset audit
│   ├── PHASE_1_REPORT.md    # Phase 1 verification report
│   ├── PHASE_2_REPORT.md    # Phase 2 dataset & model training report
│   ├── PHASE_2_1_AUDIT.md   # Independent audit of Phase 2
│   ├── PHASE_3_REPORT.md    # Phase 3 real-time webcam system report
│   └── PROJECT_STATUS.md    # Milestones & setup log
├── models/                  # Saved Keras models (git-ignored)
│   └── best_model.keras     # Checkpointed weights (28k params)
├── reports/                 # Confusion matrices, ROC curves, evaluation reports
│   ├── figures/             # Evaluation charts & confusion matrix
│   ├── dataset_analysis.md  # Exploratory data analysis report
│   ├── evaluation_metrics.json
│   └── evaluation_report.md # Model benchmark evaluation report
├── src/
│   ├── core/                # Landmark normalization & MediaPipe abstraction
│   │   ├── normalization.py # 126-D dual-hand spatial normalization
│   │   └── hand_detector.py # MediaPipe Hands dual-hand detection wrapper
│   ├── data/                # Data extraction, validation, splitting, augmentation
│   ├── inference/           # Real-time inference engine & stabilization
│   │   ├── benchmark.py     # Hardware latency profiling script
│   │   ├── gesture_state.py # 4-state hold FSM & single-character emission
│   │   ├── predictor.py     # Low-latency direct tensor predictor
│   │   ├── smoothing.py     # Confidence-aware rolling window smoother
│   │   ├── text_builder.py  # Interactive fingerspelling text buffer
│   │   └── webcam.py        # Webcam manager with handedness alignment
│   ├── models/              # MLP architecture, training, and evaluation
│   └── ui/                  # User Interfaces
│       └── opencv_app.py    # Real-time OpenCV HUD with progress bar & hotkeys
├── tests/                   # Pytest automated test suite (26 tests)
│   ├── test_normalization.py# Mathematical invariance tests
│   ├── test_phase2.py       # MLP shapes, augmentation, top-k tests
│   ├── test_phase3.py       # Predictor, smoother, state machine, text tests
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

# Activate the conda environment
conda activate ai_sign_lang
```

### 2. Run Automated Test Suite
```bash
python -m pytest tests -v
```

### 3. Launch Real-Time Webcam Application
```bash
python src/ui/opencv_app.py --camera 0 --conf_thresh 0.70 --hold_frames 12
```

**HUD Controls:**
- `C` / `c`: Clear text buffer.
- `B` / `b`: Backspace.
- `SPACE`: Add a word space.
- `Q` / `q` or `ESC`: Exit.

### 4. Run Hardware Latency Benchmark
```bash
python src/inference/benchmark.py --frames 60
```

---

## 📜 License
MIT License. Created for MCA Graduate Portfolio & Technical Viva.
