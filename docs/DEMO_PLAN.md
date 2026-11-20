# Video Demonstration Plan & Storyboard

**Project:** Real-Time Indian Sign Language (ISL) Recognition System V2  
**Target Duration:** 2 Minutes 30 Seconds  
**Format:** Screen Recording (Desktop GUI) + Webcam Picture-in-Picture  
**Goal:** Showcase genuine real-time recognition, stabilization, text construction, and engineering rigor for recruiters and interviewers.  

---

## 1. Scene-by-Scene Storyboard

### Scene 1: Introduction & Problem Statement (0:00 – 0:30)
- **Visual:** Project title screen transitioning to the launched Desktop Application (`python run_app.py`). Clean dark mode interface visible with live camera preview.
- **Narration:**  
  *"Hello! I'm presenting my MCA portfolio project: A Real-Time Indian Sign Language Recognition System. Indian Sign Language enables communication for millions of deaf and hard-of-hearing individuals, but digital communication tools often lack support for ISL fingerspelling. My goal was to build a low-latency, reliable system that runs smoothly on standard consumer CPUs without requiring an expensive GPU."*

### Scene 2: Live Sign Recognition & HUD Overview (0:30 – 1:10)
- **Visual:** Presenter raises hand and signs digits `1`, `2`, `3` and letters `A`, `B`, `C`.
  - Highlight the 52pt prediction typography.
  - Highlight the 5-state pill transitioning: `IDLE` $\to$ `DETECTING` $\to$ `HOLDING` $\to$ `COMMITTED` (green).
  - Point to the Top-3 hypotheses ranking and live FPS counter (~14–30 FPS).
- **Narration:**  
  *"Here you see the system recognizing alphanumeric gestures in real time. We extract 21 3D joint landmarks using MediaPipe, apply a custom 126-dimensional translation- and scale-invariant spatial normalizer, and pass the vector to a 28,000-parameter deep MLP. Notice the state badge: as I hold a sign steady for 0.8 seconds, the hold progress bar fills and transitions to 'COMMITTED', emitting exactly one character."*

### Scene 3: Sentence Construction & Editing Controls (1:10 – 1:45)
- **Visual:** Presenter spells the word `HELLO`, hits `[Space]`, then spells `AI`.
  - Demonstrates `[Backspace]` to fix a character.
  - Clicks `[Copy to Clipboard]` and pastes into Notepad.
- **Narration:**  
  *"To enable practical communication, the system includes a complete sentence assembly manager. As characters commit, they assemble into current words and full sentences. Notice how the state machine locks after committing to prevent accidental duplicate characters. I can insert spaces between words, backspace to correct mistakes, or copy the entire text to the system clipboard with a single click."*

### Scene 4: Real-Time Tuning Controls & Engineering Integrity (1:45 – 2:15)
- **Visual:** Presenter navigates to the 'Tuning Controls' tab.
  - Adjusts the Confidence Threshold slider from 0.70 to 0.80.
  - Adjusts the Commit Hold Frames slider from 12 to 8 frames.
  - Toggles the Hand Mesh overlay off and on.
  - Switches to 'Supported Signs & Notes' tab, showing the explicit notice that digit 0 is absent.
- **Narration:**  
  *"Every user has different signing speeds and camera conditions. Our interface lets users adjust the confidence threshold and hold duration in real time with thread-safe parameter locks. We also maintain strict engineering honesty: digit 0 was not present in the original dataset, so rather than fabricating fake data, it is documented as unsupported."*

### Scene 5: Conclusion & Technical Architecture Summary (2:15 – 2:30)
- **Visual:** Slide showing latency breakdown (0.31 ms MLP inference, 29 pytest tests passing) and GitHub repository link.
- **Narration:**  
  *"In total, our deep learning forward pass takes just 0.31 milliseconds on CPU, backed by 29 automated unit and integration tests. The full code, technical architecture, and model card are available on my GitHub. Thank you!"*
