# Sign Language Recognition V2 — Project Status & Setup Log

## 1. Project Locations
- **Original Project (Preserved / Untouched):** `D:\project\Ai_sign language`
- **New V2 Project:** `D:\project\sign_language_v2`

## 2. Phase 0 Completion Status: SUCCESSFUL
Phase 0 (Workspace Cleanup and Environment Setup) has established an independent, production-grade workspace following the architecture detailed in `SIGN_LANGUAGE_V2_BLUEPRINT.md`.

## 3. Environment Details
- **Host OS:** Windows (x64)
- **Active Base Python:** Python 3.13.9 (`D:\Anaconda3\python.exe`)
- **Recommended V2 Python:** **Python 3.10.20**
  - Confirmed available in pre-existing conda environment: `D:\Anaconda3\envs\ai_sign_lang`
  - Python 3.10 is the verified gold standard for MediaPipe 0.10.x, OpenCV 4.x, and TensorFlow 2.16–2.19, avoiding Python 3.12/3.13 C-extension and API removal issues.
- **Git Version:** `git version 2.53.0.windows.1`
- **Available Disk Space (D: Drive):** 15.42 GB free (out of 100.01 GB)

## 4. Directory Structure Created
```text
D:\project\sign_language_v2\
├── config\                    # Centralized configuration (YAML/Pydantic)
├── data\
│   ├── raw\                   # Subject-tagged raw landmark recordings
│   └── processed\             # Normalized, leakage-free train/val/test splits
├── docs\                      # Technical documentation & tracking
│   └── PROJECT_STATUS.md      # This tracking document
├── models\                    # Checkpointed weights and exported ONNX models
├── notebooks\                 # Data analysis and model benchmarking
├── reports\
│   └── figures\               # Confusion matrices, ROC curves, loss plots
├── src\                       # Modular application source code
│   ├── core\                  # Landmark normalization & MediaPipe abstraction
│   ├── data\                  # Guided collection & 3D coordinate augmentation
│   ├── inference\             # Low-latency threaded engine, state machine, TTS
│   ├── models\                # MLP architecture, training, and evaluation
│   └── ui\                    # OpenCV HUD and Streamlit web dashboard
├── tests\                     # Pytest suite (invariance tests, missing hand tests)
├── .gitignore                 # Strict policy excluding large datasets and models
├── README.md                  # Professional portfolio documentation
└── requirements.txt           # Pinned production dependencies
```

## 5. Preserved Original Assets (Backup Manifest)
The original project (`D:\project\Ai_sign language`) remains completely untouched:
- `SIGN_LANGUAGE_PROJECT_AUDIT.md`: Comprehensive audit report preserved.
- `SIGN_LANGUAGE_V2_BLUEPRINT.md`: Full V2 architecture specification preserved.
- `models/best_model.h5` (7.28 MB) & `action.h5` (7.23 MB): Original trained models intact.
- `MP_Data/` (44,100 `.npy` files across 49 classes): Original landmark dataset intact.
- `archive.zip` (294.5 MB, 42,745 Indian Sign Language images): Intact, not duplicated.
- `files/` nested repository: Git history and remote (`https://github.com/Aditya-err/Ai-Sign-Lang.git`) intact.
- `Logs/` (13 TensorBoard event files): Complete training history preserved.
- Source scripts (`main.py`, `src/*.py`, `webview_app.py`): All preserved for reference.

## 6. Dependency Installation Plan (For Phase 1)
No packages were installed during Phase 0. When Phase 1 begins, the environment can be activated or created via:
```bash
# Option A: Activate existing verified Python 3.10 environment
conda activate ai_sign_lang

# Option B: Create a fresh isolated conda environment
conda create -n sign_v2 python=3.10 -y
conda activate sign_v2
pip install -r requirements.txt
```

## 7. Remaining Setup Tasks
- Phase 0 Complete: Workspace & Environment Setup.
- Awaiting user approval to proceed to Phase 1: Core Normalization, Guided Data Collection, and Dataset Preparation.
