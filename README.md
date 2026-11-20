# Sign Language Recognition V2 — Real-Time Spatial Translation Engine

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16%2B-orange.svg)](https://tensorflow.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Hands-teal.svg)](https://mediapipe.dev)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An industry-standard, high-performance Sign Language Recognition system built with **MediaPipe Hands**, deterministic **spatial landmark normalization**, an **optimized Deep MLP classifier**, and an **event-driven finite state machine** for stable real-time fingerspelling to speech translation.

---

## 🚀 Key Improvements in Version 2

- **Zero-Noise Coordinate Pipeline:** Stripped 1,404 facial mesh points present in legacy versions; focuses exclusively on 21 hand landmarks ($63$ features).
- **Translation & Scale Invariance:** Implements wrist-relative translation ($\mathbf{p}_i - \mathbf{p}_{\text{wrist}}$) and palm-scale normalization ($\|\mathbf{p}_{\text{middle\_mcp}} - \mathbf{p}_{\text{wrist}}\|_2$), making recognition invariant to camera distance and user position.
- **Architectural Efficiency:** Replaced a 599,000-parameter over-parameterized LSTM with an optimized 20,000-parameter Deep MLP running at $< 1\text{ ms}$ latency on standard CPUs ($> 35\text{ FPS}$).
- **State Machine Debounce:** Eliminates prediction chattering with a 5-state gesture hold engine (`IDLE` $\to$ `DETECTING` $\to$ `HOLDING` $\to$ `COMMITTED` $\to$ `COOLDOWN`).
- **Data Leakage-Proof:** Evaluated using Participant-Independent (GroupKFold / Leave-One-Subject-Out) protocols.

---

## 📁 Project Structure

```text
sign_language_v2/
├── config/                  # Centralized YAML configuration files
├── data/
│   ├── raw/                 # Participant-tagged landmark JSON files
│   └── processed/           # Leakage-free train/val/test splits
├── models/                  # Checkpointed weights and exported ONNX models
├── notebooks/               # Exploratory analysis & model benchmarking
├── reports/                 # Confusion matrices, ROC curves, evaluation reports
├── src/
│   ├── core/                # Landmark normalization & MediaPipe abstraction
│   ├── data/                # Data collection & 3D coordinate augmentation
│   ├── models/              # MLP architecture, training, and evaluation
│   ├── inference/           # Threaded engine, state machine, and TTS
│   └── ui/                  # OpenCV HUD and Streamlit web dashboard
├── tests/                   # Pytest test suite (invariance tests, missing hand tests)
├── docs/                    # Technical documentation & project status
└── requirements.txt         # Pinned production dependencies
```

---

## 🛠️ Environment & Setup

Recommended runtime: **Python 3.10**

```bash
# Clone the repository
git clone <repo-url>
cd sign_language_v2

# Create virtual environment
conda create -n sign_v2 python=3.10 -y
conda activate sign_v2

# Install dependencies
pip install -r requirements.txt
```

---

## 📊 Target Classes (36 Alphanumeric Classes)

- **Alphabets:** `A` through `Z` (26 classes)
- **Digits:** `0` through `9` (10 classes)

---

## 📜 Development Status

See [docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md) for live milestone tracking and roadmap progression.
