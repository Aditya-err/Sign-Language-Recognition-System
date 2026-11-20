"""
Application Launcher for Indian Sign Language Recognition System V2.
=====================================================================

Usage:
------
1. Launch Desktop GUI (Recommended):
   python run_app.py

2. Launch Lightweight OpenCV HUD Mode:
   python run_app.py --opencv

3. Specify Camera Device Index:
   python run_app.py --camera 1
"""

import argparse
import sys
from pathlib import Path

# Ensure repo root is on sys.path
repo_root = Path(__file__).resolve().parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


def main():
    parser = argparse.ArgumentParser(
        description="Real-Time Indian Sign Language Recognition System V2"
    )
    parser.add_argument(
        "--opencv",
        action="store_true",
        help="Launch the lightweight OpenCV HUD window instead of the Tkinter desktop GUI.",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam device index (default: 0).",
    )
    args = parser.parse_args()

    if args.opencv:
        print("[INFO] Launching OpenCV HUD Application...")
        from src.ui.opencv_app import SignLanguageApp as OpenCVApp

        app = OpenCVApp(camera_index=args.camera)
        app.run()
    else:
        print("[INFO] Launching Desktop GUI Application...")
        from src.ui.desktop_app import SignLanguageApp

        app = SignLanguageApp()
        app.mainloop()


if __name__ == "__main__":
    main()
