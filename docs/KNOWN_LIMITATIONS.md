# Known Limitations & Engineering Boundaries

**Project:** Real-Time Indian Sign Language (ISL) Recognition System V2  
**Document:** Technical Limitations & System Boundaries  
**Version:** 2.0.0  

---

## 1. Executive Disclosure

In professional engineering and research, clearly defining what a system *cannot* do is just as important as demonstrating what it *can* do. This document outlines the documented technical boundaries of the ISL V2 system.

---

## 2. Documented Technical Limitations

### 2.1 Absence of Digit '0'
- **Description:** Digit `0` cannot be predicted by this model.
- **Root Cause:** The underlying source dataset (`archive.zip`) only contained classes 1–9 and A–Z.
- **Remediation:** Rather than fabricating fake data or mixing incompatible datasets, the system officially discloses that digit 0 is unsupported. A data collection script (`src/data/live_recorder.py`) has been deployed to collect genuine samples in future iterations.

### 2.2 Static Fingerspelling vs Continuous Sign Language
- **Description:** This system recognizes isolated, static alphanumeric hand poses. It does **not** recognize full continuous sign language sentences.
- **Technical Context:** Continuous sign language involves:
  - Dynamic trajectory motions across 3D space.
  - Non-manual markers (facial expressions, head tilts, mouthing).
  - Spatial grammar and contextual signs (e.g., words, phrases, idioms).
- **Engineering Justification:** Recognizing fingerspelled letters and digits is an essential foundation that enables spelling names, technical terms, and numbers, while remaining computationally lightweight on edge devices.

### 2.3 Single RGB Camera Depth Ambiguity
- **Description:** MediaPipe estimates 3D landmark coordinates ($z$-axis relative to wrist) from a single 2D RGB camera frame using monocular depth estimation.
- **Boundary:** Landmarks that are severely occluded (e.g. fingers tucked behind the palm or one hand placed directly behind the other) can suffer from depth estimation ambiguity.

### 2.4 Environmental Sensitivity (Illumination & Background)
- **Extreme Low-Light (<50 lux):** Camera sensor noise and slow shutter speeds introduce motion blur, causing MediaPipe to lose palm tracking.
- **Complex Skin-Tone Backgrounds:** While landmark normalization is skin-color invariant, the initial BlazePalm detector in MediaPipe relies on visual contrast. Highly cluttered backgrounds containing wood grains or skin-colored objects may occasionally delay hand acquisition.

### 2.5 Frame-Level Split Autocorrelation
- **Description:** The 99.61% test accuracy achieved in Phase 2 represents a stratified random partition of video frames from the original dataset.
- **Boundary:** Because consecutive frames in video datasets exhibit spatial correlation, this score should not be cited as cross-subject generalization. Real-world performance on new signers in new environments must be evaluated independently.

---

## 3. Future Development Roadmap (Phase 8+)

1. **Phase 8.1: Digit 0 Data Collection**: Use `src/data/live_recorder.py` with multiple participants to collect genuine ISL digit '0' samples and retrain the 36-class classifier.
2. **Phase 8.2: Sequence Modeling for Dynamic Signs**: Integrate a lightweight 1D-CNN or Temporal Convolutional Network (TCN) operating on sliding windows of landmark trajectories to recognize dynamic words (e.g. "HELLO", "THANK YOU").
3. **Phase 8.3: LLM Contextual Autocomplete**: Integrate an offline lightweight language model to provide smart word completion and spell correction on top of the text buffer.
