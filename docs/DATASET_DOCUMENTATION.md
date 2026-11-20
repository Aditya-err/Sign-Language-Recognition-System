# Dataset Documentation & Data Pipeline Specification

**Project:** Real-Time Indian Sign Language (ISL) Recognition System V2  
**Document:** Dataset Architecture & Feature Engineering Pipeline  
**Version:** 2.0.0  

---

## 1. Dataset Origin & Physical Structure

The dataset for this project was derived from an archive of Indian Sign Language (ISL) gesture images collected for static alphabet and numerical fingerspelling recognition.

### 1.1 Class Breakdown (35 Classes Total)
- **Digits (9 Classes):** `1, 2, 3, 4, 5, 6, 7, 8, 9`
- **Alphabets (26 Classes):** `A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z`
- **Class Balance:** Approximately 145–150 samples per class, totaling 5,100 extracted frames.

### 1.2 The Absence of Digit '0'
- **Root Cause:** An exhaustive audit of the source dataset revealed that digit `0` was entirely absent from the original archive.
- **Academic & Engineering Decision:** Rather than synthesizing fake data or borrowing incompatible American Sign Language (ASL) '0' images, we officially classified digit `0` as **unsupported** and clearly documented this boundary across all user interfaces and reports.
- **Future Resolution:** Planned for Phase 8 using our dedicated real-world recording tool (`src/data/live_recorder.py`).

---

## 2. Feature Extraction Pipeline: MediaPipe 3D Landmarks

Raw RGB video frames ($640 \times 480 \times 3$) are converted into structured geometric coordinates using **MediaPipe Hands** (`mediapipe.solutions.hands`).

```
[Raw Camera Image]
        │
        ▼
[MediaPipe Hands Pipeline]
  ├── Palm Detector (BlazePalm Single-Shot Detector)
  └── Hand Landmark Model (21 3D Joint Keypoints per Hand)
        │
        ├── Left Hand:  21 Landmarks (x, y, z)
        └── Right Hand: 21 Landmarks (x, y, z)
```

### MediaPipe Landmark Schema (21 Joint Points per Hand):
- **0**: WRIST (Anchor / Origin point)
- **1–4**: THUMB (CMC, MCP, IP, TIP)
- **5–8**: INDEX FINGER (MCP, PIP, DIP, TIP)
- **9–12**: MIDDLE FINGER (MCP, PIP, DIP, TIP)
- **13–16**: RING FINGER (MCP, PIP, DIP, TIP)
- **17–20**: PINKY (MCP, PIP, DIP, TIP)

---

## 3. Coordinate Normalization Engine

Each 3D landmark is transformed to ensure mathematical invariance to translation and scale:

1. **Wrist Origin Centering**:
   $$\mathbf{P}'_i = \mathbf{P}_i - \mathbf{P}_{\text{wrist}}$$
   Ensures that where the hand appears on the camera screen does not affect the feature representation.

2. **Bounding Box Scale Normalization**:
   $$s = \max\left(\max_i x'_i - \min_i x'_i, \; \max_i y'_i - \min_i y'_i, \; \max_i z'_i - \min_i z'_i, \; 10^{-6}\right)$$
   $$\mathbf{P}''_i = \frac{\mathbf{P}'_i}{s}$$
   Ensures that moving closer to or farther from the camera does not alter the feature vector.

3. **Dual-Hand 126-D Vector Construction**:
   $$\mathbf{V}_{\text{final}} = \left[ \mathbf{P}''_{\text{left}, 0 \dots 20}, \; \mathbf{P}''_{\text{right}, 0 \dots 20} \right] \in \mathbb{R}^{126}$$
   - Indices 0–62: Left Hand (63 features)
   - Indices 63–125: Right Hand (63 features)
   - Absent hands are explicitly represented by $\mathbf{0}_{63}$.

---

## 4. Dataset Splitting & Integrity Safeguards

To prevent data contamination and guarantee reproducible evaluation:

```
Total Processed Samples (5,100)
├── Training Set:    3,570 samples (70.0%)
├── Validation Set:    765 samples (15.0%)
└── Held-Out Test:     765 samples (15.0%)
```

- **Stratified Partitioning**: Every class maintains an exact 70/15/15 representation across splits.
- **Seed Pinning**: `random_state=42` used uniformly.
- **Leakage Prevention**: No sample appears across multiple partitions. Test data was isolated and never seen during training or hyperparameter optimization.

### Note on Frame Autocorrelation:
In video frame extraction, adjacent frames exhibit spatial correlation. An independent audit revealed that 23.8% of test samples had a nearest-neighbor Euclidean distance $d < 0.05$ to a training sample. Consequently, test split accuracy (99.61%) measures frame-level classification accuracy, while generalization across unseen users must be validated independently using our live recording framework.
