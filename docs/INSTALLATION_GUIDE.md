# Installation & Setup Guide

**Project:** Real-Time Indian Sign Language (ISL) Recognition System V2  
**Platform:** Windows 10/11, Ubuntu Linux, macOS  
**Python Version:** Python 3.10 (Recommended)  

---

## 1. System Requirements

### Hardware Requirements:
- **Processor:** Dual-core 2.0 GHz CPU or better (Intel Core i3/i5/i7 or AMD Ryzen).
- **RAM:** 4 GB RAM minimum (8 GB recommended).
- **Camera:** Standard 720p or 1080p USB webcam or integrated laptop camera.
- **GPU:** Optional. Pipeline is fully optimized for CPU execution.

### Software Prerequisites:
- Python 3.10 (tested and verified on 3.10.20)
- Anaconda / Miniconda (strongly recommended for clean virtual environment isolation)
- Git (for repository management)

---

## 2. Step-by-Step Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/<your-username>/sign_language_v2.git
cd sign_language_v2
```

### Step 2: Create and Activate Virtual Environment
```bash
# Using Conda
conda create -n isl_env python=3.10 -y
conda activate isl_env
```

### Step 3: Install Core Dependencies
Install the verified dependencies from `requirements.txt`:
```bash
pip install -r requirements.txt
```

Core dependencies installed:
- `tensorflow>=2.19.0`
- `mediapipe>=0.10.14`
- `opencv-python>=4.10.0`
- `numpy>=1.26.4`
- `scikit-learn>=1.5.0`
- `matplotlib>=3.9.0`
- `Pillow>=10.0.0`
- `pytest>=8.0.0`

---

## 3. Verification & Diagnostic Tests

### Step 1: Run the Automated Pytest Suite
Run the full test suite (29 tests covering normalization, predictor, smoother, FSM, and UI thread safety):
```bash
python -m pytest tests -v
```
Expected output:
```
============================= 29 passed in 19.92s =============================
```

### Step 2: Run the Hardware Latency Benchmark
Profile your system's camera and inference loop:
```bash
python src/inference/benchmark_full.py --frames 60
```
This confirms that camera capture, MediaPipe detection, normalization, and TensorFlow direct tensor execution are functioning correctly on your hardware.

---

## 4. Running the Application

### Option A: Launch the Desktop Graphical Application (Recommended)
```bash
python run_app.py
```

### Option B: Launch the Lightweight OpenCV HUD
```bash
python run_app.py --opencv
```

### Option C: Select a Secondary Webcam (Index 1)
```bash
python run_app.py --camera 1
```

---

## 5. Troubleshooting & FAQ

### Q1: `cv2.VideoCapture(0)` fails or camera feed is black
- Verify that your webcam is not locked by another application (Zoom, Teams, Chrome).
- In Windows Settings $\to$ Privacy & Security $\to$ Camera, ensure "Let desktop apps access your camera" is toggled ON.
- Try specifying `--camera 1` if you have multiple cameras or a virtual camera installed.

### Q2: MediaPipe warning `IMAGE_DIMENSIONS` or `NORM_RECT`
- This is a known, benign warning from the MediaPipe XNNPACK C++ delegate on Windows. It does not affect tracking accuracy or frame rates.

### Q3: Low FPS or High Latency on older laptops
- Ensure the laptop is plugged into wall power (battery saver throttles CPU clock speeds).
- In the desktop application, you can toggle off "Draw Hand Mesh" to reduce visual overlay render overhead.
