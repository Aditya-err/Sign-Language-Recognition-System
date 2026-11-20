# Dataset Analysis & Feature Extraction Report

**Date:** 2026-11-20  
**Pipeline:** MediaPipe Hands (Static Mode) + Dual-Hand Normalization  
**Feature Vector Length:** 126 floats (63 Left + 63 Right)  

## 1. Executive Summary

- **Total Images Scanned:** 5250  
- **Successfully Extracted Samples:** 5100  
- **Failed / Unusable Detections:** 150  
- **Overall Detection Success Rate:** 97.14%  
- **Number of Active Classes:** 35 (Digits 1-9 and Letters A-Z)  
- **Average Samples per Class:** 145.7 (min: 65, max: 150)  

## 2. Hand Morphology & ISL Grammar Analysis

A foundational finding of this analysis is that Indian Sign Language is fundamentally **bimodal**:
- **Dual-Handed Signs:** 14 out of 26 letters (e.g. A, B, D, E, G, M, N, P, S, T, W, X, Y, Z) predominantly involve both hands acting in coordination.
- **Single-Handed Signs:** Digits 1 through 9 and letters like C, I, L, O, V utilize a single active hand (either left or right depending on the signer's handedness).

| Metric | Total Count | Proportion |
| :--- | :--- | :--- |
| **Two Hands Detected (Dual)** | 1740 | 34.1% |
| **Left Hand Only** | 1154 | 22.6% |
| **Right Hand Only** | 2206 | 43.3% |

![Class Distribution](figures/class_distribution.png)

![Hand Types](figures/hand_types.png)

## 3. Geometric Feature Normalization Statistics

Landmarks are centered at the wrist (0, 0, 0) and scaled by palm length (Euclidean distance from wrist to middle MCP):
- **Minimum non-zero coordinate:** -2.7126
- **Maximum non-zero coordinate:** 2.4662
- **Mean non-zero coordinate:** -0.2882
- **Standard deviation:** 0.6693
- **Finite check:** All feature values are strictly finite (0.0 NaN or Inf values detected).

## 4. Per-Class Breakdown Table

| Class | Total Samples | Dual Hands | Left Hand | Right Hand | Primary Mode |
| :---: | :---: | :---: | :---: | :---: | :--- |
| `1` | 150 | 0 (0.0%) | 0 | 150 | Single-Handed |
| `2` | 150 | 0 (0.0%) | 0 | 150 | Single-Handed |
| `3` | 150 | 0 (0.0%) | 0 | 150 | Single-Handed |
| `4` | 150 | 0 (0.0%) | 0 | 150 | Single-Handed |
| `5` | 150 | 0 (0.0%) | 0 | 150 | Single-Handed |
| `6` | 150 | 0 (0.0%) | 0 | 150 | Single-Handed |
| `7` | 150 | 0 (0.0%) | 0 | 150 | Single-Handed |
| `8` | 150 | 0 (0.0%) | 0 | 150 | Single-Handed |
| `9` | 150 | 0 (0.0%) | 1 | 149 | Single-Handed |
| `A` | 150 | 150 (100.0%) | 0 | 0 | Two-Handed |
| `B` | 150 | 81 (54.0%) | 69 | 0 | Two-Handed |
| `C` | 117 | 0 (0.0%) | 67 | 50 | Single-Handed |
| `D` | 150 | 150 (100.0%) | 0 | 0 | Two-Handed |
| `E` | 150 | 150 (100.0%) | 0 | 0 | Two-Handed |
| `F` | 150 | 63 (42.0%) | 2 | 85 | Single-Handed |
| `G` | 150 | 146 (97.3%) | 1 | 3 | Two-Handed |
| `H` | 150 | 0 (0.0%) | 0 | 150 | Single-Handed |
| `I` | 150 | 0 (0.0%) | 103 | 47 | Single-Handed |
| `J` | 150 | 0 (0.0%) | 150 | 0 | Single-Handed |
| `K` | 150 | 144 (96.0%) | 0 | 6 | Two-Handed |
| `L` | 150 | 0 (0.0%) | 0 | 150 | Single-Handed |
| `M` | 150 | 149 (99.3%) | 1 | 0 | Two-Handed |
| `N` | 150 | 146 (97.3%) | 2 | 2 | Two-Handed |
| `O` | 125 | 0 (0.0%) | 7 | 118 | Single-Handed |
| `P` | 150 | 1 (0.7%) | 149 | 0 | Single-Handed |
| `Q` | 65 | 1 (1.5%) | 64 | 0 | Single-Handed |
| `R` | 150 | 58 (38.7%) | 92 | 0 | Single-Handed |
| `S` | 148 | 62 (41.9%) | 28 | 58 | Single-Handed |
| `T` | 150 | 130 (86.7%) | 20 | 0 | Two-Handed |
| `U` | 150 | 1 (0.7%) | 149 | 0 | Single-Handed |
| `V` | 145 | 0 (0.0%) | 61 | 84 | Single-Handed |
| `W` | 150 | 0 (0.0%) | 131 | 19 | Single-Handed |
| `X` | 150 | 150 (100.0%) | 0 | 0 | Two-Handed |
| `Y` | 150 | 93 (62.0%) | 57 | 0 | Two-Handed |
| `Z` | 150 | 65 (43.3%) | 0 | 85 | Single-Handed |

## 5. Technical Takeaways for Interview Discussion

1. **Why not pad missing hands with arbitrary numbers?**  
   Padding with arbitrary numbers (or facial landmarks as done in V1) contaminates the feature space.
   Using exact 0.0 for absent hands allows the network's ReLU activations to naturally zero out weights corresponding to the absent hand without corrupting gradient backpropagation.
2. **Why both Left and Right slots?**  
   Allowing separate Left and Right slots ensures that two-handed signs have distinct semantic representations for active and base hands.
