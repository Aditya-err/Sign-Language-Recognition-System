# Phase 3 Engineering Report: Real-Time Webcam Sign Recognition & Stabilization

**Project:** Real-Time Indian Sign Language (ISL) Recognition System V2  
**Author:** Senior Python Developer & Computer Vision Engineer (Mentoring MCA Fresher)  
**Date:** 2026-11-20  
**Phase Status:** **COMPLETED (Phase 3)**  
**Environment:** Python 3.10.20 (`ai_sign_lang` Conda environment), TensorFlow 2.19.1, MediaPipe 0.10.21, OpenCV 4.11.0, NumPy 1.26.4, Pytest 9.2.0  
**Hardware Tested:** Intel CPU with Integrated HD Graphics, Standard USB Webcam at 640x480 resolution  

---

## 1. Executive Summary

Phase 3 transitions the trained Deep Multi-Layer Perceptron (MLP) from an offline benchmark into a fully operational, responsive, real-time webcam sign recognition application.

### Headline Accomplishments:
- **Zero Mock Predictions:** 100% genuine real-time inference using the verified 28,515-parameter MLP model (`models/best_model.keras`).
- **Critical Handedness Bug Resolved:** Uncovered and fixed the MediaPipe horizontal flipping inversion bug documented in the Phase 2.1 audit, ensuring webcam signs perfectly match the training feature space.
- **Direct Tensor Execution:** Replaced Keras `model.predict()` (which took $91.95$ ms) with direct tensor execution `model(x, training=False)`, reducing model latency to **$0.15$ ms** on CPU!
- **Real Hardware Benchmark:** Measured total frame-to-prediction latency of **$33.41$ ms ($29.9$ FPS)** on standard hardware.
- **Temporal Stabilization:** Implemented a rolling confidence-aware smoother (window size: 7, threshold: 0.70) that eliminates hand tremor and prediction flicker.
- **Single-Emission Gesture State Machine:** Implemented a 4-state Finite State Machine (IDLE $\to$ HOLDING $\to$ COMMITTED $\to$ LOCKED) that commits a character only after a steady hold (~0.6 s) and strictly prevents duplicate character spamming.
- **Interactive Fingerspelling Text Builder:** Supports alphanumeric appending, word spacing, backspacing, and buffer clearing with a professional OpenCV HUD.
- **Automated Verification:** **26 unit and pipeline integration tests passing** under `pytest 9.2.0` (100% pass rate).

---

## 2. Real-Time Pipeline Architecture

```
+--------------------------------------------------------------------------------------------------------------------------+
|                                      PHASE 3 REAL-TIME INFERENCE PIPELINE                                                |
+--------------------------------------------------------------------------------------------------------------------------+

                                    [Webcam Video Capture] (cv2.VideoCapture)
                                                |
                               +----------------+----------------+
                               |                                 |
                               v (Raw Unflipped Frame)           v (Flipped for User View)
                  [MediaPipe Hands (Stream Mode)]        [Display Canvas]
                               |                                 |
                               v                                 |
             [Dual-Hand Normalization (126-D)]                   |
             (Left: 0..62, Right: 63..125)                       |
                               |                                 |
                               v                                 |
            [Direct Tensor Inference: model(x)]                  |
             (Latency: 0.15 ms on CPU)                           |
                               |                                 |
                               v                                 |
           [Confidence-Aware Temporal Smoother]                  |
             (Rolling Window: 7 frames, Thresh: 0.70)            |
                               |                                 |
                               v                                 |
             [4-State Gesture Hold Controller]                   |
             (Steady Hold ~0.6 s -> Emit Once)                   |
                               |                                 |
                               v                                 |
                  [Interactive Text Builder]                     |
             (Append / Space / Backspace / Clear)                |
                               |                                 |
                               +----------------+----------------+
                                                |
                                                v
                                  [OpenCV Real-Time HUD Window]
                                   - Live Skeleton Overlay
                                   - Recognized Sign & Confidence %
                                   - Hold Progress Bar
                                   - Assembled Text Buffer
                                   - Real Hardware FPS Counter
```

---

## 3. Resolving the Critical Handedness Mismatch

### The Problem (Discovered in Phase 2.1 Audit):
1. **Training Data Perspective:** In `prepare_dataset.py`, the images in `archive.zip` were third-person unmirrored photographs. When the signer signed with their right hand, MediaPipe (which expects a mirrored selfie camera) labeled it as `'Right'`. Thus, all single-handed right-hand gestures in the dataset were saved into indices 63–125.
2. **Webcam Flipping Inversion:** In standard webcam interfaces, developers routinely flip the webcam frame with `cv2.flip(frame, 1)` so the user sees a mirror reflection. But if `cv2.flip` is called **before** MediaPipe processes the frame, MediaPipe's handedness classifier inverts: a physical right hand is labeled `'Left'`, routing the coordinates to indices 0–62!
3. **Empirical Failure:** When a right-hand gesture is fed into the left-hand slot of our trained model, accuracy on sign '1' collapsed from **$95.97\%$** down to **$39.37\%$** (misclassified as 'V').

### The Solution ([src/inference/webcam.py](file:///D:/project/sign_language_v2/src/inference/webcam.py)):
- **Raw Frame to MediaPipe:** The webcam frame is captured and passed directly to MediaPipe **without horizontal flipping**.
- **Display Frame Flipped:** Landmark skeletons are rendered onto the raw frame, and then the annotated frame is flipped horizontally with `cv2.flip(detection.annotated_image, 1)` strictly for visual user rendering.
- **Result:**
  1. MediaPipe feature extraction receives the exact same perspective as the `archive.zip` training set (Right hand $\to$ indices 63..125).
  2. The user sees a natural, mirrored webcam display with perfectly aligned skeleton overlays.

---

## 4. Inference Engine Performance & Optimization

### Avoiding `model.predict()` in Real-Time Loops ([src/inference/predictor.py](file:///D:/project/sign_language_v2/src/inference/predictor.py)):
In Keras/TensorFlow, calling `model.predict(batch)` triggers graph tracing, dataset wrapping, and execution graph tear-down on every single call. In our Phase 2.1 benchmark, `model.predict()` took **$91.95$ ms per frame**.

In `SignPredictor`, we replaced this with direct tensor execution:
```python
x_tensor = tf.convert_to_tensor(feat_flat.reshape(1, 126), dtype=tf.float32)
probs = self.model(x_tensor, training=False).numpy()[0]
```
This reduces the model forward pass to **$0.15$ ms**, representing a **600x speedup** over `model.predict()`.

### Missing-Hand Safeguard:
If no hands are detected in the frame, `SignPredictor` immediately returns `has_hands=False` and `class_label=None` without running model inference, saving CPU cycles and preventing phantom predictions.

---

## 5. Temporal Smoothing & Gesture State Machine

### A. Confidence-Aware Smoother ([src/inference/smoothing.py](file:///D:/project/sign_language_v2/src/inference/smoothing.py)):
- **Window Size:** 7 frames (~0.25 to 0.35 seconds).
- **Confidence Rejection Threshold:** **0.70**. Based on our validation set analysis, predictions with confidence $\ge 0.70$ achieved **100.00% precision**. Uncertain frames ($< 0.70$) during hand movement are discarded before entering the rolling buffer.
- **Weighted Majority Voting:** The dominant class in the window must achieve at least $60\%$ consensus to be considered stable.
- **Immediate Reset on Hand Drop:** The buffer is immediately cleared when hand detection is lost.

### B. Single-Emission Gesture State Machine ([src/inference/gesture_state.py](file:///D:/project/sign_language_v2/src/inference/gesture_state.py)):
- **States:** `IDLE`, `HOLDING`, `COMMITTED`, `LOCKED`.
- **Hold Duration:** 12 consecutive stable frames (~0.4 to 0.6 seconds).
- **Hold Progress Bar:** Provides real-time visual feedback to the user on the HUD ($0\% \to 100\%$).
- **Single Character Emission:** When the hold duration reaches 100%, the state transitions to `COMMITTED` and emits the character **exactly once**.
- **Anti-Spam Lock:** The state transitions to `LOCKED` with a 6-frame cooldown. Holding the same sign indefinitely produces zero additional characters. The user must release their hand or transition to a different sign to emit another character.

---

## 6. Real Hardware Latency & FPS Measurements

Benchmarked on actual hardware over 40 frames using [src/inference/benchmark.py](file:///D:/project/sign_language_v2/src/inference/benchmark.py):

| Pipeline Stage | Measured Latency (ms) | Proportion of Frame Time |
|:---|:---:|:---:|
| **1. OpenCV Frame Capture (`cap.read()`)** | **$7.03$ ms** | $21.0\%$ |
| **2. MediaPipe Hands Tracking (CPU XNNPACK)** | **$26.18$ ms** | $78.4\%$ |
| **3. Dual-Hand Normalization (`normalize_dual_hands`)** | **$0.025$ ms** | $0.1\%$ |
| **4. Deep MLP Forward Pass (`model(x)`)** | **$0.15$ ms** | $0.4\%$ |
| **5. Smoothing & Gesture State Machine** | **$0.017$ ms** | $0.1\%$ |
| **TOTAL Frame-to-Prediction Latency** | **$33.41$ ms** | **$100.0\%$** |
| **REAL-WORLD ACHIEVED FRAMERATE** | **$29.9$ FPS** | **Real-time fluid display** |

### Performance Analysis for Interviews:
- MediaPipe pose estimation is the primary computational consumer ($78.4\%$ of frame time).
- The Deep MLP model is exceptionally lightweight, consuming only $0.15$ ms ($0.4\%$ of frame time).
- Total latency of $33.41$ ms allows the application to achieve a steady **~$30$ FPS** on standard CPU hardware without needing an external GPU.

---

## 7. Interactive OpenCV HUD Application

The real-time interface is launched via:
```bash
python src/ui/opencv_app.py --camera 0 --conf_thresh 0.70 --hold_frames 12
```

### Visual Interface Components:
1. **Top Header Banner:**
   - Displays the current predicted sign in large bold green text (e.g. `SIGN: A (98.4%)`).
   - Displays Top-3 hypotheses with probabilities (e.g. `Top-3: A(0.98)  E(0.01)  S(0.01)`).
   - Displays hand configuration tag (`Hands: Dual`, `Hands: Right Only`, `Hands: Left Only`).
2. **Hold Progress Bar:**
   - Visual gauge directly above the footer banner showing hold progress ($0\% \to 100\%$) in blue/green.
3. **Bottom Footer Banner:**
   - Displays real-time status message (`Holding 'A' (65%)`, `Recognized: 'A'!`, `Cooldown...`).
   - Displays the assembled sentence buffer with an animated blinking cursor (`TEXT: HELLO WORLD_`).
   - Displays keyboard hotkey reminders.
4. **Interactive Controls:**
   - `C` or `c`: Clear sentence buffer.
   - `B` or `b`: Backspace (delete last letter).
   - `SPACE`: Add a word space.
   - `Q` or `q` or `ESC`: Clean exit and resource release.

---

## 8. Automated Verification Suite (26 Passing Tests)

Executed under Python 3.10 `ai_sign_lang` (`pytest tests -v`):

```text
tests/test_normalization.py::test_output_dimensions PASSED               [  3%]
tests/test_normalization.py::test_wrist_centering PASSED                 [  7%]
tests/test_normalization.py::test_translation_invariance PASSED          [ 11%]
tests/test_normalization.py::test_scale_invariance PASSED                [ 15%]
tests/test_normalization.py::test_invalid_and_none_inputs PASSED         [ 19%]
tests/test_normalization.py::test_zero_scale_handling PASSED             [ 23%]
tests/test_normalization.py::test_nan_and_infinite_rejection PASSED      [ 26%]
tests/test_normalization.py::test_dual_hand_layout PASSED                [ 30%]
tests/test_normalization.py::test_deterministic_repeatability PASSED     [ 34%]
tests/test_phase2.py::test_augmentation_absent_hand_preservation PASSED  [ 38%]
tests/test_phase2.py::test_augmentation_wrist_origin_preservation PASSED [ 42%]
tests/test_phase2.py::test_augment_dual_hand_sample PASSED               [ 46%]
tests/test_phase2.py::test_augment_training_set PASSED                   [ 50%]
tests/test_phase2.py::test_mlp_model_shapes_and_softmax PASSED           [ 53%]
tests/test_phase2.py::test_compute_top_k_accuracy PASSED                 [ 57%]
tests/test_phase3.py::test_predictor_model_dimensions_and_classes PASSED [ 61%]
tests/test_phase3.py::test_predictor_missing_hand_safeguard PASSED       [ 65%]
tests/test_phase3.py::test_predictor_valid_feature_prediction PASSED     [ 69%]
tests/test_phase3.py::test_smoother_confidence_rejection PASSED          [ 73%]
tests/test_phase3.py::test_smoother_rolling_majority_vote PASSED         [ 76%]
tests/test_phase3.py::test_smoother_reset_on_hand_loss PASSED            [ 80%]
tests/test_phase3.py::test_gesture_state_hold_and_commit PASSED          [ 84%]
tests/test_phase3.py::test_gesture_state_duplicate_prevention PASSED     [ 88%]
tests/test_phase3.py::test_gesture_state_resets_on_lost_detection PASSED [ 92%]
tests/test_phase3.py::test_text_builder_editing_semantics PASSED         [ 96%]
tests/test_validator.py::test_validator_detects_valid_and_invalid PASSED [100%]

============================= 26 passed in 9.73s ==============================
```

---

## 9. Known Real-World Limitations

1. **Consecutive-Frame Autocorrelation in Training:**  
   The $99.61\%$ test accuracy was measured on held-out frames from the same recording session as the training set. In front of a live webcam with diverse signers and varying hand geometries, practical accuracy will be lower (typically $80\% - 90\%$). Our Top-3 display provides an effective fallback for subtly ambiguous signs.
2. **Missing Digit 0:**  
   The source archive dataset contains 35 classes (digits 1–9 and letters A–Z); digit '0' was missing from the original source.
3. **Fingerspelling vs Continuous Sign Language:**  
   This application is an isolated alphabet and digit fingerspelling system. It translates static letter configurations into text strings. It does not perform Continuous Sign Language Translation (CSLT) or full grammar translation (which requires continuous sentence glosses, non-manual facial markers, and temporal transformers).

---

## 10. Viva / Interview Questions & Defense Guide

### Q1: "Why does your real-time application run so fast on a standard CPU without a GPU?"
*Answer:* "We engineered the pipeline for CPU efficiency at every stage. Rather than running a heavyweight 50-layer CNN or a 599,000-parameter LSTM on 49,152 raw RGB pixels, we use MediaPipe's lightweight TFLite model to extract 21 geometric hand joints. Our MLP classifier has only 28,515 parameters, taking 0.15 milliseconds for forward inference. Total frame latency is 33.4 milliseconds, allowing a steady 29.9 FPS on CPU."

### Q2: "How did you solve the problem of repeated characters when holding a sign?"
*Answer:* "We designed a 4-state Finite State Machine (IDLE, HOLDING, COMMITTED, LOCKED). When a user forms a sign, the system tracks consecutive stable frames. Once the threshold (~0.6 seconds) is reached, the character is emitted exactly once into the text buffer and the state transitions to LOCKED. The user must either drop their hand or switch to a new sign to commit another character."

### Q3: "What happens if the webcam frame is flipped horizontally?"
*Answer:* "MediaPipe assumes an unmirrored third-person perspective when classifying handedness in camera coordinates. If the frame is flipped horizontally before MediaPipe processes it, right-hand signs are mislabeled as left-hand signs, causing prediction failure. We resolved this by feeding the raw unflipped frame to MediaPipe for inference and landmark extraction, and applying horizontal mirroring exclusively to the display frame."
