# Real-World Testing Protocol: Indian Sign Language Recognition V2

**Document Version:** 1.0.0  
**Project:** Real-Time Indian Sign Language (ISL) Recognition System V2  
**Target Architecture:** MediaPipe Hands + Dual-Hand Normalizer (126-D) + Deep MLP (28k params)  
**Status:** PROTOCOL ACTIVE / PHYSICAL VALIDATION SESSIONS PENDING PARTICIPANT EXECUTION

---

## 1. Executive Summary & Philosophy of Testing

In accordance with strict academic and engineering integrity standards:
1. **Never conflate benchmark split accuracy with real-world accuracy**: The 99.61% accuracy achieved in Phase 2 represents frame-level classification on a stratified random split of the dataset. Because temporal video frames exhibit spatial autocorrelation, real-world generalization across new participants, unseen lighting, and webcam hardware variations must be evaluated independently.
2. **Unsupported Classes**: Digit `0` is absent from the underlying dataset and is officially unsupported. Any presentation or documentation must explicitly state this boundary.
3. **No Fabricated Testing**: Physical webcam testing requires actual human execution in front of a live camera. Tests that have not been conducted with human signers are explicitly marked as **`STATUS: PENDING`**.

---

## 2. Supported Gestures Inventory (35 Classes)

| Category | Gestures | Total Classes | Hand Configuration |
| :--- | :--- | :--- | :--- |
| **Digits (1–9)** | `1, 2, 3, 4, 5, 6, 7, 8, 9` | 9 | Single hand (dominant) |
| **Alphabets (A–Z)** | `A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z` | 26 | Mixed: Single hand (e.g., C, I, L, O, V) & Dual hand (e.g., A, B, D, P, W) |
| **Unsupported** | `0` (Zero) | 0 | Not present in training dataset |

---

## 3. Environmental Testing Matrix

To rigorously quantify real-world robustness, evaluations must span four key environmental variables:

### 3.1 Illumination Matrix
- **Condition L1 (Optimal Daylight):** Diffuse natural daylight (300–500 lux), indirect window light, front-facing.
- **Condition L2 (Low Indoor Lighting):** Warm domestic LED lighting (<100 lux), slight shadow fall across fingers.
- **Condition L3 (High-Contrast / Backlit):** Light source behind subject causing silhouette effect on hand contours.
- **Condition L4 (Overhead Harsh Fluorescent):** Standard institutional / office ceiling light (cold white).

### 3.2 Background Complexity Matrix
- **Condition B1 (Controlled / Neutral):** Plain solid white, grey, or off-white wall. Zero moving objects.
- **Condition B2 (Cluttered Domestic / Office):** Bookshelves, furniture, posters, clothing textures in camera frustum.
- **Condition B3 (Dynamic / Distractor):** Distant ambient motion, secondary persons passing in the background.

### 3.3 Subject Distance & Angle Matrix
- **Condition D1 (Close-Up / Desktop):** 0.5 meters from webcam. Hands occupy ~40–60% of vertical frame height.
- **Condition D2 (Nominal Operational):** 0.8–1.2 meters from webcam. Torso and upper chest visible.
- **Condition D3 (Far / Edge Boundary):** 1.8–2.2 meters from webcam. Hands occupy <15% of vertical frame height.
- **Condition A1 (Direct Frontal):** Camera at eye/chest level, perpendicular ($0^\circ$).
- **Condition A2 (Elevation Offset):** Laptop webcam tilted up from desk level ($+15^\circ$ to $+25^\circ$).

### 3.4 Morphological Variation
- Hand sizes: Small, Medium, Large hand spans.
- Skin pigmentation: Varying Fitzpatrick scale skin types to verify MediaPipe skin-invariance.
- Left-dominant vs Right-dominant signers.

---

## 4. Standard Operational Procedure (SOP) for Human Evaluator

### 4.1 Calibration & Camera Alignment
1. Position webcam so that the user's upper torso and both hands are clearly within the 640x480 frame.
2. Ensure ambient illumination is adequate (MediaPipe hand confidence threshold $\ge 0.5$).
3. Keep hands relaxed below camera view before beginning each gesture trial.

### 4.2 Single-Trial Lifecycle
```
[Hands at Rest (Out of View)]
         │
         ▼
[Form Gesture & Raise into Camera View]
         │
         ▼
[Hold Gesture Steady for 2.0 Seconds] ───► Observe Stability & Confidence
         │
         ▼
[Check State Machine Trigger] ──────────► Verify Word/Sentence Buffer Append
         │
         ▼
[Lower Hands Below View for 1.0 Second] ─► State Machine Resets to IDLE
```

### 4.3 Evaluation Metrics per Class
For each class $c \in \{1..9, A..Z\}$ over $N=10$ trials:
- **True Positive (TP):** Correct sign recognized with confidence $\ge 0.70$ and committed to text buffer.
- **False Positive (FP):** Incorrect sign recognized or sign triggered prematurely.
- **False Negative (FN):** Sign held for 2 seconds but no recognition triggered (confidence $< 0.70$ or undetected).
- **Latency to Commit:** Frames elapsed between gesture formation and buffer append.
- **Flicker Index:** Number of label oscillations between raw consecutive frames before smoothing.

---

## 5. Structured 35-Class Verification Checklist

| Class | Expected Hands | Test Condition | Target Recall | Observed Result | Status |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | Single (Dominant) | L1, B1, D2 | $\ge 90\%$ | *Pending live session* | **STATUS: PENDING** |
| **2** | Single (Dominant) | L1, B1, D2 | $\ge 90\%$ | *Pending live session* | **STATUS: PENDING** |
| **3** | Single (Dominant) | L1, B1, D2 | $\ge 90\%$ | *Pending live session* | **STATUS: PENDING** |
| **4** | Single (Dominant) | L1, B1, D2 | $\ge 90\%$ | *Pending live session* | **STATUS: PENDING** |
| **5** | Single (Dominant) | L1, B1, D2 | $\ge 90\%$ | *Pending live session* | **STATUS: PENDING** |
| **6** | Single (Dominant) | L1, B1, D2 | $\ge 90\%$ | *Pending live session* | **STATUS: PENDING** |
| **7** | Single (Dominant) | L1, B1, D2 | $\ge 90\%$ | *Pending live session* | **STATUS: PENDING** |
| **8** | Single (Dominant) | L1, B1, D2 | $\ge 90\%$ | *Pending live session* | **STATUS: PENDING** |
| **9** | Single (Dominant) | L1, B1, D2 | $\ge 90\%$ | *Pending live session* | **STATUS: PENDING** |
| **A** | Dual Hand | L1, B1, D2 | $\ge 85\%$ | *Pending live session* | **STATUS: PENDING** |
| **B** | Dual Hand | L1, B1, D2 | $\ge 85\%$ | *Pending live session* | **STATUS: PENDING** |
| **C** | Single Hand | L1, B1, D2 | $\ge 90\%$ | *Pending live session* | **STATUS: PENDING** |
| **D** | Dual Hand | L1, B1, D2 | $\ge 85\%$ | *Pending live session* | **STATUS: PENDING** |
| **E** | Dual Hand | L1, B1, D2 | $\ge 85\%$ | *Pending live session* | **STATUS: PENDING** |
| **F** | Dual Hand | L1, B1, D2 | $\ge 85\%$ | *Pending live session* | **STATUS: PENDING** |
| **G** | Dual Hand | L1, B1, D2 | $\ge 85\%$ | *Pending live session* | **STATUS: PENDING** |
| **H** | Dual Hand | L1, B1, D2 | $\ge 85\%$ | *Pending live session* | **STATUS: PENDING** |
| **I** | Single Hand | L1, B1, D2 | $\ge 90\%$ | *Pending live session* | **STATUS: PENDING** |
| **J** | Dual Hand | L1, B1, D2 | $\ge 80\%$ | *Pending live session* | **STATUS: PENDING** |
| **K** | Dual Hand | L1, B1, D2 | $\ge 85\%$ | *Pending live session* | **STATUS: PENDING** |
| **L** | Single Hand | L1, B1, D2 | $\ge 90\%$ | *Pending live session* | **STATUS: PENDING** |
| **M** | Dual Hand | L1, B1, D2 | $\ge 85\%$ | *Pending live session* | **STATUS: PENDING** |
| **N** | Dual Hand | L1, B1, D2 | $\ge 85\%$ | *Pending live session* | **STATUS: PENDING** |
| **O** | Single Hand | L1, B1, D2 | $\ge 90\%$ | *Pending live session* | **STATUS: PENDING** |
| **P** | Dual Hand | L1, B1, D2 | $\ge 85\%$ | *Pending live session* | **STATUS: PENDING** |
| **Q** | Dual Hand | L1, B1, D2 | $\ge 85\%$ | *Pending live session* | **STATUS: PENDING** |
| **R** | Dual Hand | L1, B1, D2 | $\ge 85\%$ | *Pending live session* | **STATUS: PENDING** |
| **S** | Dual Hand | L1, B1, D2 | $\ge 85\%$ | *Pending live session* | **STATUS: PENDING** |
| **T** | Dual Hand | L1, B1, D2 | $\ge 85\%$ | *Pending live session* | **STATUS: PENDING** |
| **U** | Single / Dual | L1, B1, D2 | $\ge 85\%$ | *Pending live session* | **STATUS: PENDING** |
| **V** | Single Hand | L1, B1, D2 | $\ge 90\%$ | *Pending live session* | **STATUS: PENDING** |
| **W** | Dual Hand | L1, B1, D2 | $\ge 85\%$ | *Pending live session* | **STATUS: PENDING** |
| **X** | Dual Hand | L1, B1, D2 | $\ge 85\%$ | *Pending live session* | **STATUS: PENDING** |
| **Y** | Single Hand | L1, B1, D2 | $\ge 90\%$ | *Pending live session* | **STATUS: PENDING** |
| **Z** | Dual Hand | L1, B1, D2 | $\ge 85\%$ | *Pending live session* | **STATUS: PENDING** |

---

## 6. Execution Command for User Validation

The dedicated CLI verification runner is ready in `src/inference/live_validation.py`:

```bash
# Run the interactive live validation protocol across all 35 classes
python src/inference/live_validation.py --trials-per-class 5
```

Results are automatically saved to `reports/live_validation_results.json` without modifying any baseline benchmark datasets.
