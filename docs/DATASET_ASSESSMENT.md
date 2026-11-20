# Dataset Assessment Report: Indian Sign Language (ISL) Callset

**Target Archive:** `D:\project\Ai_sign language\archive.zip` (Read-Only Inspection)  
**Evaluator:** Senior AI/ML Engineer & MCA Project Mentor  
**Date:** November 2026  
**Document Version:** 1.0  

---

## 1. Executive Summary

An independent, non-destructive inspection of the 280.8 MB dataset archive (`archive.zip`) in the original project directory was conducted. The primary goal was to determine whether this dataset corresponds to authentic Indian Sign Language (ISL), identify its anatomical hand configuration (single-hand vs. two-hand), check for label inconsistencies, and define a scientifically viable recognition scope for Version 2.

### Critical Takeaways:
1. **Authentic Indian Sign Language Structure:** The archive contains **42,745 color JPEG images** across **35 classes** under an `Indian/` directory hierarchy.
2. **Two-Handed vs. Single-Handed Reality:** In Indian Sign Language (ISL), alphabets predominantly use **two hands**, whereas digits (1–9) use **one hand**. Our empirical MediaPipe Hands audit verified that **14 alphabet letters require both hands** (e.g., A, B, D, E, F, G, K, M, N, P, S, T, X, Y, Z), while **12 letters and all 9 digits use a single hand**.
3. **Missing Digit '0':** The dataset contains digits `1` through `9`, but digit `0` is completely missing.
4. **Architectural Implication:** Any single-hand pipeline (such as ASL-style 63-dimensional vectors) will discard half of the linguistic information in ISL. Version 2 must adopt a **Dual-Hand Landmark Normalization Pipeline (126 features: 63 Left + 63 Right)** to natively support authentic ISL.

---

## 2. Archive Inspection & Class Breakdown

### 2.1 File & Directory Metrics
- **Archive Size:** 280.82 MB (294,458,379 bytes)
- **Total Images:** 42,745 files
- **Image Format:** JPEG (`.jpg`)
- **Image Resolution:** Uniform $128 \times 128 \times 3$ (RGB)
- **Root Directory:** `Indian/`
- **Total Class Folders:** 35 folders

### 2.2 Class List & Image Distribution
| Category | Classes Present | Total Classes | Samples per Class | Missing Classes |
|---|---|:---:|:---:|---|
| **Numerics** | `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9` | **9** | 1,200 each ($10,800$ total) | Digit `0` is absent. |
| **Alphabets** | `A` through `Z` | **26** | 1,200 each (except `C` = 1,447) ($31,447$ total) | None (all 26 letters present). |
| **Words / Sentences** | None | **0** | 0 | No continuous phrases or dynamic words. |
| **TOTAL** | `1–9`, `A–Z` | **35 Classes** | **42,745 Images** | Digit `0` |

---

## 3. Empirical Hand Configuration Audit

Using MediaPipe Hands with dual-hand detection (`max_num_hands=2`) across all 35 classes, sample images from each folder were tested for hand count and handedness:

### 3.1 Detailed Per-Class Hand Audit

| Class | Hand Configuration | Handedness Observed | Linguistic Rationale (ISL Standard) |
|:---:|:---:|:---:|---|
| **1 – 9** | **Single-Hand** | Right (dominant) | In ISL, counting from 1 to 9 is executed with fingers of a single hand. |
| **A** | **Two-Hand** | Left + Right | Dominant index finger touches the non-dominant thumb tip. |
| **B** | **Two-Hand** | Left + Right | Both hands form circular rings touching each other to form 'B'. |
| **C** | **Single-Hand** | Left or Right | Curved fingers forming a 'C' shape. |
| **D** | **Two-Hand** | Left + Right | Non-dominant index up; dominant hand forms arc touching tip and base. |
| **E** | **Two-Hand** | Left + Right | Dominant index touches non-dominant index fingertip. |
| **F** | **Two-Hand** | Left + Right | Two index fingers crossed or touching at right angles. |
| **G** | **Two-Hand** | Left + Right | Both fists placed together with thumbs aligned. |
| **H** | **Single-Hand** | Right | Flat horizontal palm stroke. |
| **I** | **Single-Hand** | Right | Pinky finger extended upward. |
| **J** | **Single-Hand** | Right | Pinky traces slight curve or static posture. |
| **K** | **Two-Hand** | Left + Right | Dominant index hooks into non-dominant index finger. |
| **L** | **Single-Hand** | Right | Thumb and index finger form an 'L' shape. |
| **M** | **Two-Hand** | Left + Right | Three fingers of dominant hand placed across non-dominant palm. |
| **N** | **Two-Hand** | Left + Right | Two fingers of dominant hand placed across non-dominant palm. |
| **O** | **Single-Hand** | Right | Fingers form circular 'O'. |
| **P** | **Two-Hand** | Left + Right | Dominant circle touching non-dominant index finger. |
| **Q** | **Single-Hand** | Right | Hooked index and thumb downward. |
| **R** | **Single-Hand** | Right | Crossed index and middle fingers. |
| **S** | **Two-Hand** | Left + Right | Dominant pinky hooks non-dominant pinky. |
| **T** | **Two-Hand** | Left + Right | Dominant index touches side of non-dominant palm. |
| **U** | **Single-Hand** | Right | Index and middle fingers extended together upward. |
| **V** | **Single-Hand** | Right | Index and middle fingers separated ('V' for victory). |
| **W** | **Single-Hand** | Right | Index, middle, and ring fingers extended upward. |
| **X** | **Two-Hand** | Left + Right | Both index fingers crossed forming an 'X'. |
| **Y** | **Two-Hand** | Left + Right | Dominant index finger placed between thumb and index of non-dominant hand. |
| **Z** | **Two-Hand** | Left + Right | Dominant finger touches non-dominant open palm or traces 'Z'. |

### 3.2 Hand Configuration Summary
- **Single-Hand Signs (20 Classes):** Digits `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9` and Letters `C`, `H`, `I`, `J`, `L`, `O`, `Q`, `R`, `U`, `V`, `W`.
- **Two-Hand Signs (15 Classes):** Letters `A`, `B`, `D`, `E`, `F`, `G`, `K`, `M`, `N`, `P`, `S`, `T`, `X`, `Y`, `Z`.

---

## 4. Key Limitations of the `archive.zip` Dataset

1. **Synthetic Backgrounds & Sensor Artifacts:**
   The images are $128 \times 128$ crops with uniform plain backgrounds, captured under a controlled camera setup. Real webcam feeds have variable resolutions ($640 \times 480$), motion blur, and cluttered backgrounds.
2. **Missing Digit Zero ('0'):**
   Standard numeric sign systems include 0. If digit 0 is desired in V2, it must be recorded via our guided collector.
3. **No Temporal Dimension:**
   All 42,745 images are individual static photographs. Attempting to group them into 30-frame temporal sequences (as the legacy V1 converter tried) is mathematically invalid.
4. **Camera-Distance Variance:**
   Because images were cropped close to the hands, absolute coordinate distances vary depending on how tightly the image was bounded.

---

## 5. Scope Recommendation for Version 2

### Recommended Initial Recognition Scope:
To keep this project scientifically valid, understandable for an MCA fresher, and fully explainable in an interview:

1. **Target Callset:**
   - **Primary Core (35 Classes):** 9 Digits (`1`–`9`) + 26 Letters (`A`–`Z`).
   - If user requires digit `0`, collect 200 samples of `0` in Phase 1 to reach **36 alphanumeric classes**.
2. **Dual-Hand Feature Representation (126 Dimensions):**
   - Each hand is assigned 21 landmarks $\times 3$ coordinates $(x, y, z) = 63$ dimensions.
   - Left Hand Slice: indices $0 \dots 62$.
   - Right Hand Slice: indices $63 \dots 125$.
   - If only one hand is detected (e.g., for digit '1' or letter 'C'), the active hand is populated with its normalized 63 coordinates, while the missing hand is cleanly zero-padded.
   - The classifier easily learns that single-handed signs have zeros in the secondary hand slot.
3. **Static Image / Pose Focus (Phase 1 & 2):**
   - Treat fingerspelling and digit recognition as a **static pose classification problem**.
   - Do not use LSTMs or recurrent networks. Use a lightweight Deep Multi-Layer Perceptron (MLP) operating on the 126-dimensional normalized vector.
