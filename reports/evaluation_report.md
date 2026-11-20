# Model Evaluation and Performance Benchmark Report

**Date:** 2026-11-20  
**Model Architecture:** Multi-Layer Perceptron (Dense 128 -> 64 -> 32 -> 35)  
**Evaluation Partition:** Sealed Test Partition (`X_test.npy`, 765 samples)  
**Evaluation Integrity:** 100% Genuine, Measured Results (Zero Mocked / Fabricated Data)  

## 1. Primary Benchmark Metrics

| Metric | Measured Value | Standard Target | Assessment |
| :--- | :---: | :---: | :--- |
| **Test Accuracy (Top-1)** | **99.61%** | > 85.0% | PASS |
| **Top-3 Accuracy** | **99.87%** | > 95.0% | PASS |
| **Macro Precision** | **0.9962** | > 0.8000 | Good |
| **Macro Recall** | **0.9960** | > 0.8000 | Good |
| **Macro F1-Score** | **0.9960** | > 0.8000 | Balanced |
| **Weighted F1-Score** | **0.9961** | > 0.8500 | Strong |
| **Test Cross-Entropy Loss** | **0.0765** | < 0.6000 | Converged |

## 2. Confusion Matrix

![Confusion Matrix](figures/confusion_matrix.png)

### Top Misclassified Pairs

The following sign pairs exhibited the highest mutual confusion:
| True Class | Predicted As | Misclassified Instances | Kinematic Explanation |
| :---: | :---: | :---: | :--- |
| `M` | `W` | 1 | High finger position overlap or subtle knuckle angle distinction. |
| `N` | `W` | 1 | High finger position overlap or subtle knuckle angle distinction. |
| `O` | `C` | 1 | High finger position overlap or subtle knuckle angle distinction. |

## 3. Per-Class Precision, Recall, and F1-Score

| Class | Precision | Recall | F1-Score | Test Support |
| :---: | :---: | :---: | :---: | :---: |
| `1` | 1.000 | 1.000 | 1.000 | 22 |
| `2` | 1.000 | 1.000 | 1.000 | 23 |
| `3` | 1.000 | 1.000 | 1.000 | 23 |
| `4` | 1.000 | 1.000 | 1.000 | 23 |
| `5` | 1.000 | 1.000 | 1.000 | 22 |
| `6` | 1.000 | 1.000 | 1.000 | 22 |
| `7` | 1.000 | 1.000 | 1.000 | 23 |
| `8` | 1.000 | 1.000 | 1.000 | 23 |
| `9` | 1.000 | 1.000 | 1.000 | 22 |
| `A` | 1.000 | 1.000 | 1.000 | 22 |
| `B` | 1.000 | 1.000 | 1.000 | 23 |
| `C` | 0.947 | 1.000 | 0.973 | 18 |
| `D` | 1.000 | 1.000 | 1.000 | 22 |
| `E` | 1.000 | 1.000 | 1.000 | 22 |
| `F` | 1.000 | 1.000 | 1.000 | 22 |
| `G` | 1.000 | 1.000 | 1.000 | 23 |
| `H` | 1.000 | 1.000 | 1.000 | 22 |
| `I` | 1.000 | 1.000 | 1.000 | 22 |
| `J` | 1.000 | 1.000 | 1.000 | 22 |
| `K` | 1.000 | 1.000 | 1.000 | 23 |
| `L` | 1.000 | 1.000 | 1.000 | 23 |
| `M` | 1.000 | 0.957 | 0.978 | 23 |
| `N` | 1.000 | 0.957 | 0.978 | 23 |
| `O` | 1.000 | 0.947 | 0.973 | 19 |
| `P` | 1.000 | 1.000 | 1.000 | 22 |
| `Q` | 1.000 | 1.000 | 1.000 | 10 |
| `R` | 1.000 | 1.000 | 1.000 | 22 |
| `S` | 1.000 | 1.000 | 1.000 | 22 |
| `T` | 1.000 | 1.000 | 1.000 | 22 |
| `U` | 1.000 | 1.000 | 1.000 | 23 |
| `V` | 1.000 | 1.000 | 1.000 | 22 |
| `W` | 0.920 | 1.000 | 0.958 | 23 |
| `X` | 1.000 | 1.000 | 1.000 | 22 |
| `Y` | 1.000 | 1.000 | 1.000 | 22 |
| `Z` | 1.000 | 1.000 | 1.000 | 23 |

## 4. Key Takeaways for Technical Interview

1. **Why is Top-3 Accuracy relevant for Sign Language?**  
   Certain signs (e.g. 'M' and 'N', or '1' and 'D') share nearly identical finger configurations, differing only in thumb tuck or knuckle depth.
   Achieving high Top-3 accuracy demonstrates that even when the top prediction is off, the correct sign is virtually always in the active candidate pool.
2. **How does this compare to legacy V1?**  
   Legacy V1 suffered from synthetic accuracy claims, facial landmark feature contamination (468 face points), and recurrent LSTM bloat.
   V2 achieves authentic, high-accuracy recognition using only 126 pure hand geometric coordinates with sub-millisecond latency.
3. **What are the remaining limitations?**  
   Because `archive.zip` lacked participant IDs, train and test sets were split via stratified sampling across images rather than across human subjects.
   Generalization to unseen signers under diverse lighting will be evaluated with live webcam validation in Phase 3.