# System Architecture & Technical Specification

**Project:** Real-Time Indian Sign Language (ISL) Recognition System V2  
**Document:** Project Architecture Specification  
**Version:** 2.0.0  
**Target Platform:** Windows x64 / Linux / macOS (CPU-Optimized)  

---

## 1. High-Level Architectural Overview

The Real-Time Indian Sign Language Recognition System V2 transforms raw webcam video into formatted text in real time. Unlike legacy approaches that rely on heavy end-to-end 3D CNNs or unconstrained LSTM networks, this system separates **spatial geometric feature extraction** from **classification**, achieving high frame rates on standard consumer CPUs without requiring a dedicated GPU.

```
┌──────────────┐     Raw BGR Frame (640x480)
│ Video Source │ ──────────────────────────────────────┐
└──────────────┘                                       │
                                                       ▼
                                            ┌──────────────────────┐
                                            │ MediaPipe Hands      │
                                            │ (TFLite XNNPACK CPU) │
                                            └──────────┬───────────┘
                                                       │ 21 Left + 21 Right 3D Landmarks
                                                       ▼
                                            ┌──────────────────────┐
                                            │ Spatial Normalizer   │
                                            │ (126-D Dual-Hand)    │
                                            └──────────┬───────────┘
                                                       │ L2 Normalized Vector [126]
                                                       ▼
                                            ┌──────────────────────┐
                                            │ Deep MLP Classifier  │
                                            │ (TensorFlow C++ Graph)│
                                            └──────────┬───────────┘
                                                       │ Softmax Probabilities [35]
                                                       ▼
                                            ┌──────────────────────┐
                                            │ Confidence Filter    │
                                            │ & Temporal Smoother  │
                                            └──────────┬───────────┘
                                                       │ Filtered Majority Prediction
                                                       ▼
                                            ┌──────────────────────┐
                                            │ Finite State Machine │
                                            │ (4-State Hold Logic) │
                                            └──────────┬───────────┘
                                                       │ Committed Characters
                                                       ▼
                                            ┌──────────────────────┐
                                            │ Text Buffer & UI HUD │
                                            │ (Tkinter Desktop /   │
                                            │  OpenCV HUD)         │
                                            └──────────────────────┘
```

---

## 2. Mathematical Formulation of Normalization

MediaPipe provides 21 three-dimensional landmark points per hand:
$$\mathbf{P}_i = (x_i, y_i, z_i) \quad \text{for } i \in \{0, 1, \dots, 20\}$$
where $i=0$ is the wrist landmark.

To ensure **translation invariance**, the entire hand is centered relative to its wrist coordinate:
$$\mathbf{P}'_i = \mathbf{P}_i - \mathbf{P}_0 \quad \implies \quad \mathbf{P}'_0 = (0, 0, 0)$$

To ensure **scale invariance** (so that gestures made close to the camera or far away produce identical feature representations), we compute the 3D bounding box span across all 21 centered points:
$$s_x = \max_i(x'_i) - \min_i(x'_i), \quad s_y = \max_i(y'_i) - \min_i(y'_i), \quad s_z = \max_i(z'_i) - \min_i(z'_i)$$
$$s = \max(s_x, s_y, s_z)$$

If $s < \epsilon$ (where $\epsilon = 10^{-6}$ to prevent division by zero in degenerate detections), $s$ is clamped to 1.0.

The scale-normalized coordinates are then calculated as:
$$\mathbf{P}''_i = \frac{\mathbf{P}'_i}{s}$$

### Dual-Hand Concatenation Layout
Indian Sign Language utilizes both single-hand signs (e.g., Digits 1–9, letters C, I, L, O, V) and dual-hand signs (e.g., A, B, D, P, W). The normalized vectors are organized into a strict 126-dimensional feature vector:
$$\mathbf{X} = \left[ \mathbf{P}''_{\text{left}, 0 \dots 20}, \; \mathbf{P}''_{\text{right}, 0 \dots 20} \right] \in \mathbb{R}^{126}$$

- Indices `0:63`: Left hand ($21 \times 3 = 63$ coordinates).
- Indices `63:126`: Right hand ($21 \times 3 = 63$ coordinates).
- **Missing Hand Handling**: If a hand is absent from the frame, its respective 63 indices are populated with exact zeros ($\mathbf{0}_{63}$), providing an explicit geometric signal to the network that only one hand is active.

---

## 3. Deep Learning Architecture: Multi-Layer Perceptron (MLP)

The classifier operates on the 126-dimensional input vector and maps it onto 35 target classes (1–9, A–Z).

```
Input Vector [126]
       │
       ▼
Dense(128, activation='relu', kernel_initializer='he_normal')
       │
BatchNormalization()
       │
Dropout(0.30)
       │
       ▼
Dense(64, activation='relu', kernel_initializer='he_normal')
       │
BatchNormalization()
       │
Dropout(0.30)
       │
       ▼
Dense(35, activation='softmax')
```

### Parameter Breakdown:
- Layer 1 (Dense 128): $126 \times 128 + 128 = 16,256$ params
- Layer 1 (BatchNorm): $128 \times 4 = 512$ params (256 trainable, 256 non-trainable)
- Layer 2 (Dense 64): $128 \times 64 + 64 = 8,256$ params
- Layer 2 (BatchNorm): $64 \times 4 = 256$ params (128 trainable, 128 non-trainable)
- Layer 3 (Output Dense 35): $64 \times 35 + 35 = 2,275$ params
- **Total Parameters:** **28,515** (Trainable: 28,131 | Non-trainable: 384)
- **Model Checkpoint Size:** **380 KB**

### Direct Tensor Inference vs Model Predict:
In high-throughput loops, calling `model.predict(x)` invokes Keras execution wrapper overhead (~92 ms on CPU). In V2, we pass inputs directly to the underlying callable C++ graph:
```python
# Direct Tensor Execution: 0.31 ms latency (300x faster)
tensor_x = tf.convert_to_tensor(feature_vector, dtype=tf.float32)
probabilities = model(tensor_x, training=False).numpy()[0]
```

---

## 4. Real-Time Temporal Stabilization & State Machine

Raw frame-by-frame neural network predictions suffer from two major real-world issues:
1. **Transient Jitter**: Single-frame classification noise caused by camera sensor motion blur or landmark jitter.
2. **Duplicate Character Appending**: Without a state machine, holding a sign for 1 second at 30 FPS would produce 30 identical characters.

### 4.1 Temporal Majority Voting Filter (`PredictionSmoother`)
- Uses a circular buffer of window size $W = 7$ frames.
- Rejects any frame where the model confidence is below the threshold $\tau = 0.70$.
- Requires that the most frequent class in the buffer achieves a majority consensus ($\ge 4$ out of 7 frames) before changing the active hypothesis.

### 4.2 Four-State Finite State Machine (`GestureStateManager`)

```
      [ IDLE ]  ◄───────────────────────────────────────┐
          │                                              │
     Valid Sign Detected                                 │
          │                                              │
          ▼                                              │
    [ DETECTING ]                                        │
          │                                              │
     Sign Held Consistent                                │
          │                                              │
          ▼                                              │
     [ HOLDING ] ─── Hold Frames Reaches 12 ───┐        │ Hand Lost /
          ▲                                    │         │ Sign Changes
          │                                    ▼         │
          │                             [ COMMITTED ]    │
          │                                    │         │
          │                             Commit Character │
          │                                    │         │
          │                                    ▼         │
          └──────── Sign Cleared ───────── [ LOCKED ] ───┘
```

- **IDLE**: No hand detected or confidence below threshold. Hold counter is 0.
- **DETECTING**: Initial hand gesture identified; smoothing buffer filling.
- **HOLDING**: Candidate sign is held stable. Counter increments toward $T_{\text{hold}} = 12$ frames (~0.85s).
- **COMMITTED**: Threshold reached. Exactly **one** character is emitted to the text builder.
- **LOCKED**: Prevents duplicate emissions while the user continues to hold the same sign. The user must lower their hands or change signs before a new character can be registered.

---

## 5. Desktop Application Concurrency Model

The application utilizes a decoupled, thread-safe Producer-Consumer architecture using Python's native `threading` and `queue.Queue`.

- **Producer (`InferenceWorker`)**:
  - Captures video frames from `cv2.VideoCapture`.
  - Runs MediaPipe Hands landmark extraction.
  - Computes 126-D dual-hand normalization.
  - Executes deep MLP forward pass via TensorFlow C++ tensor graph.
  - Updates smoother and state machine.
  - Places structured data packets into `queue.Queue(maxsize=2)`.
- **Consumer (Tkinter Main Loop)**:
  - Invokes `root.after(15, poll_queue)` every 15 milliseconds.
  - Retrieves latest frame packet and renders RGB canvas via `PIL.ImageTk`.
  - Updates HUD widgets, confidence bars, and text buffer without any UI thread blocking.
  - Parameter changes (sliders) are protected by a mutual exclusion `threading.Lock`.
