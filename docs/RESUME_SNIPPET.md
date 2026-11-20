# Resume Snippet & ATS-Optimized Project Entry

**Target Audience:** MCA Fresher / Aspiring AI/ML Engineer / Computer Vision Developer  
**Formats Provided:** Standard Bullet Points (STAR format) & ATS Keyword Matrix  

---

## 1. Resume Section: Projects

### **Real-Time Indian Sign Language (ISL) Recognition System V2**
*AI/ML & Computer Vision Project | Python, TensorFlow, MediaPipe, OpenCV, Tkinter, Pytest*  
*(November 2026)*

- **Engineered a real-time, low-latency ISL recognition system** translating 35 static fingerspelling gestures (alphabets A–Z, digits 1–9) to text at 14–30 FPS on standard consumer CPUs without requiring a dedicated GPU.
- **Architected a 126-D geometric normalization engine** achieving mathematical translation and scale invariance on dual-hand 3D landmarks extracted via MediaPipe Hands, completely eliminating background, lighting, and skin-tone sensitivity.
- **Trained a lightweight 28.5k-parameter Deep MLP** achieving **99.61% top-1 accuracy** and **99.87% top-3 accuracy** on a stratified test split, with an ultra-fast **0.31 ms direct tensor inference pass** (300x faster than standard Keras wrapper).
- **Developed a 4-state Finite State Machine (FSM) & 7-frame temporal majority filter**, eliminating prediction flicker and duplicate character emissions, with a Pareto-optimal confidence threshold ($\tau = 0.70$) yielding 100% precision on validation.
- **Built a thread-safe desktop GUI** implementing a decoupled Producer-Consumer pattern with bounded queues to ensure zero UI freeze during camera I/O, incorporating live HUD, word assembly, and real-time parameter tuning.
- **Implemented rigorous automated verification** with 29 passing unit/integration tests and complete technical documentation (Model Card, System Architecture, Failure Mode Analysis).

---

## 2. Technical Skills Demonstrated (For Resume Skills Section)

- **Languages:** Python (Advanced), SQL
- **Computer Vision & AI/ML:** MediaPipe Hands, OpenCV, TensorFlow / Keras, Scikit-Learn, NumPy
- **System Design & Software Engineering:** Concurrency (`threading`, `queue`), Finite State Machines (FSM), Object-Oriented Design (OOP), Producer-Consumer Architecture
- **Testing & Quality Assurance:** Pytest (29 unit & integration tests), Profiling & Latency Benchmarking
- **Documentation & Methodological Rigor:** Model Cards, Data Pipeline Specifications, Failure Mode & Root Cause Analysis
