# Portfolio Project Summary: Real-Time ISL Recognition V2

**Author:** MCA Fresher / AI & Machine Learning Specialist  
**Project:** Real-Time Indian Sign Language (ISL) Recognition System V2  
**Repository:** `github.com/<username>/sign_language_v2`  

---

## 1. Elevator Pitch (30 Seconds)

> *"Indian Sign Language Recognition System V2 is a real-time, CPU-optimized computer vision application that translates 35 static ISL fingerspelling gestures (A–Z, 1–9) directly into text. By decoupling 3D landmark extraction from classification and applying a custom 126-dimensional spatial normalization algorithm, the system runs at full camera frame rates on standard laptops with a deep learning forward pass of just 0.31 milliseconds. It features a 4-state stabilization finite state machine, a full text-assembly buffer, and a thread-safe desktop GUI—achieving 99.61% accuracy on held-out test data backed by 29 automated tests."*

---

## 2. One-Page Comprehensive Project Showcase

### The Problem
Indian Sign Language (ISL) serves as the primary mode of expression for millions of deaf and hard-of-hearing individuals across South Asia. However, digital communication interfaces rarely accommodate sign-based interaction. While prior research frequently proposed heavy 3D Convolutional Neural Networks requiring enterprise GPUs or mocked predictions with synthetic overlays, real-world accessibility requires lightweight, responsive, and verifiable AI solutions that run locally on standard consumer computers.

### The Engineering Solution
I engineered an end-to-end, edge-compatible recognition system built upon four core architectural pillars:

1. **Spatial Geometric Normalization**:
   Instead of feeding bulky RGB images to a deep network, MediaPipe Hands extracts 21 3D joint landmarks per hand. A custom mathematical engine centers each hand around its wrist origin and scales it by its 3D bounding box span, outputting a 126-dimensional vector invariant to user distance, hand span, camera position, and skin tone.
2. **Lightweight Deep MLP**:
   A 3-layer neural network with Batch Normalization and Dropout totaling just 28,515 parameters was trained on a 70/15/15 stratified split of 5,100 samples. Using direct C++ tensor execution, the model evaluates in **0.31 ms**, consuming less than 0.5% of the total frame processing budget.
3. **Temporal Stabilization & State Machine**:
   To eliminate single-frame prediction flicker and duplicate character appending, predictions pass through a 7-frame rolling majority voting smoother and a 4-state Finite State Machine (`IDLE` $\to$ `DETECTING` $\to$ `HOLDING` $\to$ `COMMITTED` $\to$ `LOCKED`). Users hold a sign for 0.8 seconds to commit a character, after which the state locks until repositioning.
4. **Thread-Safe Desktop GUI**:
   Built using Python's native `tkinter` and `PIL` without heavy external framework bloat, the application isolates camera I/O and neural network computation in a background worker thread. A non-blocking queue prevents UI freezes, while interactive sliders allow live threshold tuning.

### Key Metrics & Verified Achievements
- **Test Set Accuracy:** 99.61% Top-1 / 99.87% Top-3 on 765 held-out test frames.
- **Inference Latency:** 0.31 ms (deep learning pass) / 14.1–29.9 FPS overall throughput on standard CPU.
- **Model Storage Footprint:** 380 KB (`.keras` format).
- **Test Coverage:** 29 automated unit and integration tests passing under `pytest`.
- **Engineering Integrity:** Zero mocked predictions, zero fabricated accuracy, and explicit disclosure of missing digit '0'.
