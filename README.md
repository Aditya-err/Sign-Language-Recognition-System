# Indian Sign Language (ISL) Recognition System V2

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19-orange.svg)](https://tensorflow.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Hands%200.10-teal.svg)](https://mediapipe.dev)
[![Tests](https://img.shields.io/badge/Pytest-29%20Passed-brightgreen.svg)](tests/)
[![Inference Latency](https://img.shields.io/badge/MLP%20Inference-0.31%20ms-success.svg)](reports/benchmark_300_frames.json)
[![GUI](https://img.shields.io/badge/Desktop%20UI-Tkinter%20Thread--Safe-blueviolet.svg)](src/ui/desktop_app.py)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An industry-standard, high-performance **Real-Time Indian Sign Language (ISL) Recognition System** built with **MediaPipe Hands**, a mathematically proven **126-D dual-hand spatial normalizer**, an **optimized Deep MLP classifier** (28,515 parameters, 0.31 ms latency), and a **temporal stabilization state machine** with a decoupled, thread-safe desktop user interface.

Developed as an MCA portfolio project adhering strictly to academic and engineering integrity standards.

---

## 🌟 Why Version 2? (Technical Evolution)

| Engineering Dimension | Legacy V1 Implementation | Version 2 Architecture |
| :--- | :--- | :--- |
| **Linguistic Scope** | Assumed single-hand ASL | Native **Dual-Hand Indian Sign Language (ISL)** |
| **Classifier Model** | 599,000-parameter LSTM (Overfitted) | **Lightweight Deep MLP (28,515 parameters, 380 KB)** |
| **Feature Space** | 1,662 features (84.5% irrelevant face mesh) | **126 clean features** (63 Left + 63 Right hand landmarks) |
| **Spatial Normalization** | None (Absolute canvas coordinates) | **Wrist-origin translation + 3D bounding box scale invariance** |
| **Missing-Hand Logic** | Imputed with fabricated vectors | Explicit $\mathbf{0}_{63}$ zero-padding + missing-hand bypass guard |
| **Inference Call Path** | `model.predict()` (~92 ms overhead) | **Direct C++ tensor execution `model(x)` (0.31 ms)** |
| **Real-Time FPS** | 8–12 FPS (choppy, UI freezing) | **14–30 FPS on CPU** with non-blocking threaded queue |
| **Duplicate Prevention** | None (Spammed "AAAAA" while holding) | **4-state FSM (`IDLE` $\to$ `HOLDING` $\to$ `COMMITTED` $\to$ `LOCKED`)** |
| **User Interface** | Mocked predictions & hardcoded charts | **Live Threaded Desktop GUI + OpenCV HUD Mode** |
| **Automated Testing** | Zero automated tests | **29 unit/integration tests (100% pytest pass rate)** |

---

## 📐 System Architecture & Pipeline

```text
[Raw Webcam 640x480]
         │
         ▼
[MediaPipe Hands (TFLite CPU)] ──► Extracts 21 Left + 21 Right 3D Joint Keypoints
         │
         ▼
[126-D Spatial Normalizer]     ──► Wrist-Centering + Bounding Box Scale Normalization
         │
         ▼
[Deep MLP Classifier]          ──► Direct Tensor Forward Pass (0.31 ms, 28k params)
         │
         ▼
[Temporal Majority Filter]     ──► Rolling Window (W=7), Confidence Threshold (τ=0.70)
         │
         ▼
[4-State Gesture FSM]          ──► IDLE ──► DETECTING ──► HOLDING (12f) ──► COMMITTED ──► LOCKED
         │
         ▼
[Text Assembly & Desktop GUI]  ──► Live HUD, Sentence Buffer, Space, Backspace, Clipboard
```

### Mathematical Normalization Formulation:
1. **Wrist Translation Invariance**: Centers all 21 hand landmarks relative to the wrist origin:
   $$\mathbf{P}'_i = \mathbf{P}_i - \mathbf{P}_{\text{wrist}} \quad \implies \quad \mathbf{P}'_0 = (0, 0, 0)$$
2. **Bounding Box Scale Invariance**: Scales coordinates by the 3D bounding box span:
   $$s = \max\left(\max_i x'_i - \min_i x'_i, \; \max_i y'_i - \min_i y'_i, \; \max_i z'_i - \min_i z'_i, \; 10^{-6}\right)$$
   $$\mathbf{P}''_i = \frac{\mathbf{P}'_i}{s}$$
3. **Dual-Hand 126-D Representation**:
   $$\mathbf{X} = [\mathbf{P}''_{\text{left}}, \; \mathbf{P}''_{\text{right}}] \in \mathbb{R}^{126}$$
   Single-hand gestures populate the dominant hand slot (indices 63–125 for right hand), with absent hands padded with zeros.

---

## ⚡ Hardware Benchmark Telemetry (300 Frames Profiled)

Profiled directly on host hardware (8 CPU cores, Windows 10, Python 3.10.20):

| Pipeline Stage | Average Latency | % of Frame Time | Optimization Applied |
| :--- | :--- | :--- | :--- |
| **1. Camera Frame Capture** | 2.96 ms | 4.2% | Non-blocking OpenCV `cap.read()` |
| **2. MediaPipe Hands Tracking** | 67.05 ms | 94.8% | TFLite XNNPACK multi-threaded CPU delegate |
| **3. Dual-Hand Normalization** | 0.05 ms | 0.1% | Vectorized NumPy mathematical operations |
| **4. Deep MLP Forward Pass** | **0.31 ms** | **0.4%** | Direct `tf.Tensor` C++ graph execution |
| **5. Temporal Filter, FSM & UI** | 0.32 ms | 0.5% | Thread-safe queue packet dispatch |
| **Total End-to-End Latency** | **70.70 ms** | **100.0%** | **14.1 FPS overall throughput on CPU** |

> **Key Architectural Takeaway:** The neural network classifier consumes only **0.4%** of the frame budget. Over 94% is consumed by MediaPipe palm detection. Selecting a lightweight MLP over a heavy CNN was essential for real-time CPU execution.

---

## 🎯 Model Accuracy & Empirical Validation

- **Held-Out Test Set Accuracy:** **99.61% Top-1** (762/765 correct) | **99.87% Top-3** (764/765 correct)
- **Top Confused Pairs:** `M` $\to$ `W` (1), `N` $\to$ `W` (1), `O` $\to$ `C` (1).
- **Threshold Optimization:** In all 3 error cases, confidence was $< 0.62$. Applying our operational threshold $\tau = 0.70$ achieves **100.00% precision** on accepted validation frames while maintaining **98.69% coverage**.
- **Supported Classes (35 Total):**
  - Digits: `1, 2, 3, 4, 5, 6, 7, 8, 9`
  - Alphabets: `A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z`
  - *Digit 0 Notice*: Digit `0` was absent from the source dataset and is officially classified as unsupported.

---

## 🚀 Quick Start & Installation

### 1. Prerequisites
- Python 3.10 (recommended)
- Webcam (built-in or USB)

### 2. Setup Virtual Environment
```bash
# Clone the repository
git clone https://github.com/<your-username>/sign_language_v2.git
cd sign_language_v2

# Create and activate environment
conda create -n isl_env python=3.10 -y
conda activate isl_env

# Install dependencies
pip install -r requirements.txt
```

### 3. Verify Setup (Run 29 Pytest Tests)
```bash
python -m pytest tests -v
```

### 4. Launch Application
```bash
# Launch Desktop GUI (Recommended)
python run_app.py

# Launch Lightweight OpenCV HUD Mode
python run_app.py --opencv

# Select Secondary Camera
python run_app.py --camera 1
```

---

## 💻 Desktop Application Features

- **Decoupled Concurrency:** Video capture and AI inference run in a background worker thread (`InferenceWorker`). The UI event loop never freezes.
- **Dynamic HUD:** Displays prominent predicted sign (52pt font), 5-color state badges (`IDLE`, `DETECTING`, `HOLDING`, `COMMITTED`, `LOCKED`), confidence progress bar, and Top-3 hypotheses.
- **Interactive Tuning Sliders:** Real-time adjustment of Confidence Threshold ($\tau \in [0.50, 0.95]$) and Commit Hold Frames ($T_{\text{hold}} \in [5, 25]$).
- **Text Assembly Workspace:** Live current word preview, multiline editable sentence buffer, Space, Backspace, Clear, and Copy to Clipboard.
- **Reference Guide:** In-app reference tab with all 35 supported signs and tips for optimal lighting and camera distance.

---

## 📁 Repository Structure

```text
sign_language_v2/
├── config/                  # Configuration files
├── data/
│   ├── raw/                 # Subject-tagged raw landmark records
│   ├── processed/           # Processed 126-D features & stratified splits (git-ignored)
│   └── real_world/          # Independent live recordings directory
├── docs/                    # Comprehensive Technical Documentation Suite
│   ├── PROJECT_ARCHITECTURE.md   # Mathematical specification & architecture
│   ├── MODEL_CARD.md             # Formal ML Model Card
│   ├── DATASET_DOCUMENTATION.md  # Dataset provenance & feature pipeline
│   ├── INSTALLATION_GUIDE.md     # Step-by-step developer setup
│   ├── USER_GUIDE.md             # End-user manual & operating instructions
│   ├── KNOWN_LIMITATIONS.md      # Documented system boundaries
│   ├── INTERVIEW_PREPARATION.md  # 20 detailed technical Q&As for MCA freshers
│   ├── DEMO_PLAN.md              # 2.5-minute video demonstration storyboard
│   ├── RESUME_SNIPPET.md         # ATS-friendly resume bullet points
│   ├── PORTFOLIO_SUMMARY.md      # Elevator pitch & 1-page showcase
│   ├── MODEL_COMPARISON.md       # Architecture & retraining evaluation
│   ├── REAL_WORLD_TESTING_PROTOCOL.md # 35-class testing matrix
│   └── PHASE_1_REPORT.md through PHASE_7_REPORT.md # Audit reports
├── models/                  # Checkpointed weights
│   └── best_model.keras     # 28,515-parameter production MLP (380 KB)
├── reports/                 # Benchmarks, metrics, and evaluation summaries
│   ├── benchmark_300_frames.json # Hardware latency profiling results
│   ├── threshold_optimization.json # Validation threshold trade-off analysis
│   └── evaluation_report.md      # Test split confusion matrix & metrics
├── src/
│   ├── core/                # Landmark normalization & MediaPipe wrapper
│   │   ├── normalization.py # 126-D dual-hand spatial normalizer
│   │   └── hand_detector.py # MediaPipe Hands abstraction
│   ├── data/                # Extraction, splitting, augmentation, live recorder
│   ├── inference/           # Real-time predictor, smoother, FSM, benchmarks
│   ├── models/              # MLP architecture, training, and threshold optimization
│   └── ui/                  # Desktop Tkinter GUI & OpenCV HUD
├── tests/                   # 29 automated pytest unit and integration tests
├── requirements.txt         # Verified package dependencies
└── run_app.py               # Unified application launcher
```

---

## 📖 Technical Documentation Links

- [System Architecture Specification](docs/PROJECT_ARCHITECTURE.md)
- [Formal Model Card](docs/MODEL_CARD.md)
- [Dataset Documentation & Feature Pipeline](docs/DATASET_DOCUMENTATION.md)
- [Installation Guide](docs/INSTALLATION_GUIDE.md)
- [User Guide & Operations](docs/USER_GUIDE.md)
- [Known Limitations & Roadmap](docs/KNOWN_LIMITATIONS.md)
- [MCA Interview Preparation (20 Q&As)](docs/INTERVIEW_PREPARATION.md)
- [Video Demonstration Storyboard](docs/DEMO_PLAN.md)
- [ATS Resume Snippets](docs/RESUME_SNIPPET.md)
- [Portfolio Showcase](docs/PORTFOLIO_SUMMARY.md)

---

## ⚖️ Engineering Integrity & Ethical Disclosure

1. **Academic Honesty:** No synthetic or mocked predictions. Every reported metric is backed by automated tests, hardware profiling scripts, or confusion matrices.
2. **Unsupported Classes:** Digit `0` is absent from the original dataset and is officially classified as unsupported rather than synthesized.
3. **Autocorrelation Transparency:** We disclose that 23.8% of test split frames share proximity to training frames due to video extraction. Field validation on unseen participants is tracked independently in `reports/real_world_evaluation.md`.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
