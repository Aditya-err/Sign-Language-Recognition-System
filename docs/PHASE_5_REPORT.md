# Phase 5 Technical Report: Model Improvement & Hyperparameter Optimization

**Project:** Real-Time Indian Sign Language (ISL) Recognition System V2  
**Author:** Senior AI/ML Engineer (Mentoring MCA Fresher)  
**Date:** November 2026  
**Status:** PHASE 5 COMPLETE — ARCHITECTURE VERIFIED & THRESHOLDS OPTIMIZED  

---

## 1. Executive Summary

Phase 5 addresses a critical milestone in any rigorous machine learning engineering lifecycle: determining whether model retraining is justified by empirical evidence, or whether system reliability is better achieved through pre/post-processing optimization.

Key Achievements:
1. **Systematic Failure Mode Analysis**: Investigated the 3 misclassified test samples (out of 765 total test instances) from Phase 2. Identified exact geometric and topological mechanisms causing confusion ($M \to W$, $N \to W$, $O \to C$).
2. **Empirically-Grounded Retraining Decision**: Demonstrated that retraining on the existing 5,100 samples would risk destabilizing an already converged, highly accurate model (99.61% top-1, 99.87% top-3) without resolving hand landmark ambiguities. Formally decided to preserve `models/best_model.keras`.
3. **Quantitative Threshold Optimization**: Evaluated Softmax confidence distribution across validation data, proving that confidence threshold $\tau = 0.70$ achieves **100.00% precision** on accepted samples while retaining **98.69% coverage**.
4. **Temporal Filter & Finite State Machine Validation**: Verified that rolling majority voting ($W=7$) and hold-to-commit ($T_{hold}=12$ frames) completely suppress the single-frame ambiguities identified in the failure mode analysis.

---

## 2. Failure Mode Deep Dive

In the test evaluation report (`reports/evaluation_report.md`), the 3 misclassifications were:
- `M` predicted as `W` (Softmax probability = 0.584)
- `N` predicted as `W` (Softmax probability = 0.541)
- `O` predicted as `C` (Softmax probability = 0.612)

### Root Cause Analysis:
1. **Topological Similarity**:
   - Both 'M' and 'W' involve three adjacent extended fingers. 'M' points downward, while 'W' points upward. Because our coordinate normalization centers on the wrist and scales by bounding box dimensions, vertical orientation information is partially attenuated.
   - 'N' (two fingers extended downward) is topologically adjacent to both 'M' and 'W'.
2. **Spatial Threshold Proximity**:
   - 'C' is an open hand arc; 'O' is a closed loop where the thumb and index finger touch. When an evaluator forms an open 'C' with a narrow aperture ($<10$ mm separation between fingertip landmarks 4 and 8), MediaPipe sub-pixel estimation noise can place the tips within the contact threshold.
3. **Critical Observation**:
   - In all failure instances, the model's confidence was between **0.54 and 0.62**. None exceeded 0.70.

---

## 3. Retraining Decision & Architecture Verification

### 3.1 Decision Matrix
| Evaluation Factor | Assessment | Verdict |
| :--- | :--- | :--- |
| **Model Capacity** | 28,515 parameters over 3 dense layers is well-matched to 126-D input. | Sufficient |
| **Overfitting Risk** | Train Acc: 99.89%, Val Acc: 99.74%, Test Acc: 99.61% — minimal generalization gap. | Low |
| **Data Quality Constraint**| Original archive contains static extracted images with no video trajectories. | Retraining won't add dynamic temporal cues |
| **Computational Efficiency**| 0.31 ms per inference pass consumes only 0.4% of frame budget. | Optimal |

### 3.2 Formal Conclusion
We **preserve** `models/best_model.keras` as the production model. Changing weights will not fix MediaPipe tracking noise or camera lighting. Instead, the inference pipeline handles real-world variance via temporal filtering.

---

## 4. Hyperparameter Trade-Off Summary

From `reports/threshold_optimization.json`:
- **Confidence Threshold ($\tau = 0.70$)**: Rejects transitional postures and ambiguous frames while passing 98.69% of valid gestures with 100.00% precision.
- **Smoothing Window ($W = 7$ frames)**: Requires majority consensus over ~0.47 seconds at 15 FPS, preventing transient flickers.
- **Hold Duration ($T_{hold} = 12$ frames)**: Requires ~0.85 seconds of stable hold before appending character to text buffer, eliminating accidental keystrokes.

---

## 5. Artifacts & Deliverables

- Detailed Comparison Document: `docs/MODEL_COMPARISON.md`
- Threshold Profiling Script: `src/models/optimize_thresholds.py`
- Optimization Metrics: `reports/threshold_optimization.json`
- Phase Status Updated: `docs/PROJECT_STATUS.md`

Ready to proceed to **Phase 6: Final Application UI & User Experience**.
