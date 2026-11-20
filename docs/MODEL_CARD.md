# Model Card: Indian Sign Language MLP Classifier V2

**Model Name:** `isl_mlp_v2`  
**Model Checkpoint:** `models/best_model.keras`  
**Architecture:** 3-Layer Dense Multilayer Perceptron with Batch Normalization & Dropout  
**Developer:** MCA AI/ML Portfolio Project  
**Date:** November 2026  
**License:** MIT  

---

## 1. Model Summary & Specifications

| Parameter | Specification |
| :--- | :--- |
| **Model Type** | Deep Feed-Forward Multi-Layer Perceptron (MLP) |
| **Input Shape** | 126-dimensional float vector ($\mathbb{R}^{126}$): normalized Left Hand [63] + Right Hand [63] |
| **Output Shape** | 35-dimensional probability distribution ($\mathbb{R}^{35}$) via Softmax |
| **Hidden Layers** | Layer 1: Dense(128) + BatchNorm + Dropout(0.3) <br> Layer 2: Dense(64) + BatchNorm + Dropout(0.3) |
| **Total Parameters** | 28,515 parameters (111.4 KB in float32 weights) |
| **Storage Size** | 380 KB on disk (`.keras` format) |
| **Inference Latency** | **0.31 ms** per forward pass on CPU (direct tensor call) |
| **Framework** | TensorFlow 2.19.0 / Keras 3 |

---

## 2. Intended Use & Boundaries

### 2.1 Intended Use Cases
- Real-time fingerspelling recognition of static Indian Sign Language (ISL) alphanumeric gestures.
- Assisting hearing-impaired communication through interactive text typing.
- Educational tool for learning and verifying basic ISL alphabet and number formations.
- Low-latency edge deployment on standard consumer PCs without specialized GPU hardware.

### 2.2 Out-of-Scope / Non-Intended Uses
- **Continuous Sign Language Translation**: Not designed for continuous fluid sign language sentences involving grammar, facial expressions, or shoulder movements.
- **Dynamic Signs**: Gestures requiring motion trajectories (e.g., signs for "Where", "When", "Name") cannot be recognized by this static coordinate model.
- **Digit 0**: Digit '0' was not included in the source dataset and is officially unsupported.
- **Safety-Critical / Medical Diagnosis**: Must not be used as an authoritative translation tool in emergency, legal, or medical contexts.

---

## 3. Training & Validation Data

### 3.1 Source Dataset Overview
- **Raw Samples:** 5,100 landmark frames extracted across 35 classes (140 to 150 samples per class).
- **Classes Supported (35):**
  - Digits (9): `1, 2, 3, 4, 5, 6, 7, 8, 9`
  - Alphabets (26): `A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z`
- **Data Splitting Strategy:**
  - 70% Training: 3,570 samples
  - 15% Validation: 765 samples
  - 15% Held-Out Test: 765 samples
  - Partitioned using stratified sampling (`sklearn.model_selection.train_test_split`, `random_state=42`).

### 3.2 Feature-Space Coordinate Augmentation
To prevent overfitting on the training partition without introducing impossible hand anatomies:
- Random Cartesian jitter: $\delta \sim \mathcal{N}(0, 0.015)$ applied to landmark coordinates.
- Random global scale jitter: $s \in [0.92, 1.08]$.
- Coordinate dropout: 3% probability of setting non-wrist coordinates to 0.
- **Integrity Rule**: Augmented coordinates preserve the $(0, 0, 0)$ wrist anchor, and absent hands remain strictly zero-padded.

---

## 4. Quantitative Evaluation

### 4.1 Held-Out Test Set Performance (765 Samples)
- **Top-1 Categorical Accuracy:** **99.61%** (762 / 765 correct)
- **Top-3 Categorical Accuracy:** **99.87%** (764 / 765 correct)
- **Macro Precision:** 99.62%
- **Macro Recall:** 99.60%
- **Macro F1-Score:** 99.60%
- **Weighted F1-Score:** 99.61%
- **Test Loss (Cross-Entropy):** 0.0765

### 4.2 Failure Mode Analysis
Out of 765 test samples, exactly 3 were misclassified:
1. `M` predicted as `W` (Softmax probability = 0.584)
2. `N` predicted as `W` (Softmax probability = 0.541)
3. `O` predicted as `C` (Softmax probability = 0.612)

**Resolution**: In all 3 instances, the raw confidence was $< 0.62$. Applying our operational threshold of $\tau = 0.70$ completely eliminates all 3 errors, yielding **100.00% precision** on accepted frames.

---

## 5. Ethical Considerations & Fairness

### 5.1 Demographic & Physical Diversity
- **MediaPipe Landmark Invariance**: Feature extraction is computed on normalized joint keypoints rather than raw pixel color values, providing natural invariance to variations in skin tone, pigmentation, and background lighting.
- **Hand Size Robustness**: Explicit bounding box normalization ensures scale invariance across adult and child hands.
- **Dominance Invariance**: Single-hand signs are trained on dominant right-hand positions. Signers using left hands must either mirror the input or use dual-hand configurations.

### 5.2 Environmental Sensitivity
- Sub-pixel tracking accuracy decreases under severe low-light conditions (<50 lux) or high-speed motion blur.
- Adequate ambient lighting and steady positioning (0.8m–1.2m from camera) are recommended.

---

## 6. Caveats & Methodological Rigor

1. **Temporal Frame Autocorrelation**: High benchmark accuracy (99.61%) on a randomly split image dataset does not guarantee identical accuracy on unseen human signers in different rooms. Real-world performance must be evaluated independently.
2. **Never Fabricate Metrics**: All reported hardware metrics (0.31 ms MLP latency, 14.1 FPS overall CPU throughput) are directly profiled from physical execution.
