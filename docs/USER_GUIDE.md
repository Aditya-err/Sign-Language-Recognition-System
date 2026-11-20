# User Guide & Operating Instructions

**Project:** Real-Time Indian Sign Language (ISL) Recognition System V2  
**Application:** Desktop Sign-to-Text Recognition Interface  
**Version:** 2.0.0  

---

## 1. Getting Started

When you launch the application via `python run_app.py`, the desktop graphical interface will open:

```
┌───────────────────────────────────────────────────────────────────────────────┐
│ Indian Sign Language Recognition System V2                       [FPS: 14.5]  │
├─────────────────────────────────────────┬─────────────────────────────────────┤
│                                         │ REAL-TIME RECOGNITION               │
│                                         │ ┌───────┐ State: [COMMITTED]        │
│                                         │ │   A   │ Confidence: [████████] 94%│
│                                         │ └───────┘ Commit Hold: [██████] 100%│
│          Live Webcam Feed               │ Top: [1] A (94%) [2] B (3%)         │
│             (640 x 480)                 ├─────────────────────────────────────┤
│           [Mirrored View]               │ TEXT CONVERSATION BUFFER            │
│                                         │ Current Word: HELLO                 │
│                                         │ ┌─────────────────────────────────┐ │
│                                         │ │ HELLO WORLD                     │ │
│                                         │ └─────────────────────────────────┘ │
│ [Stop Camera] [Pause AI] [x] Hand Mesh  │ [Space] [Backspace] [Clear] [Copy]  │
├─────────────────────────────────────────┴─────────────────────────────────────┤
│  Tuning: Threshold: 0.70 | Hold Frames: 12 (~0.8s)                            │
└───────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Gesture Lifecycle & Best Practices

To achieve the highest recognition accuracy:

### 2.1 Environmental Positioning
1. **Camera Alignment**: Sit 0.8 to 1.2 meters away from the webcam. Keep your torso and hands clearly visible within the frame.
2. **Lighting**: Ensure good diffuse front lighting. Avoid strong backlights (e.g. sitting with a bright window behind you).
3. **Background**: Simple, uncluttered backgrounds produce the cleanest MediaPipe landmark tracking.

### 2.2 How to Form and Commit a Character
1. **Form the Gesture**: Raise your hands into the camera view and form the desired sign (e.g., 'A', '1', 'B').
2. **Observe the HUD**:
   - The large sign label will immediately show the detected hypothesis.
   - The State badge will transition from **`IDLE`** $\to$ **`DETECTING`** $\to$ **`HOLDING`**.
3. **Hold Steady**: Maintain the hand shape for ~0.8 seconds. Watch the **Commit Hold** progress bar fill.
4. **Character Emitted**: Once the bar reaches 100%, the state transitions to **`COMMITTED`** (green), and the character is appended to your text buffer.
5. **Resetting for Next Character**: The state shifts to **`LOCKED`** (purple) to prevent duplicate letters. To type the next letter, simply relax or adjust your hand. The state resets to `IDLE` ready for the next sign.

---

## 3. Text Buffer Controls

The text construction area allows assembling full words and sentences:

| Control | Action | Keyboard / Shortcut |
| :--- | :--- | :--- |
| **Space** | Inserts a space character, completing the current word. | Click `[Space]` |
| **Backspace** | Deletes the last committed character in the buffer. | Click `[Backspace]` |
| **Clear** | Clears the entire text buffer to start fresh. | Click `[Clear]` |
| **Copy to Clipboard** | Copies the full composed sentence to the system clipboard for pasting into other apps. | Click `[Copy to Clipboard]` |

---

## 4. Interactive Tuning Controls

In the bottom panel, click the **Tuning Controls** tab to adapt to different users or environments:

- **Confidence Threshold Slider ($\tau$)**:
  - Default: `0.70`.
  - Increase (up to `0.85`) in noisy or cluttered environments to reject false triggers.
  - Decrease (down to `0.60`) if valid signs are not triggering due to weak camera lighting.
- **Commit Hold Frames ($T_{\text{hold}}$)**:
  - Default: `12 frames` (~0.8 seconds).
  - Faster typers can lower this to `8 frames` (~0.5 seconds).
  - Beginners can increase this to `16 frames` (~1.1 seconds) to allow more time to position fingers accurately.

---

## 5. Supported Gestures Reference

- **Digits (1–9)**: `1, 2, 3, 4, 5, 6, 7, 8, 9` (Single-hand dominant).
- **Alphabets (A–Z)**: `A` through `Z` (Mixed single-hand and dual-hand formations).
- **Unsupported**: Digit `0` is absent from the underlying dataset and will not be recognized.
