"""
Application Launcher for Indian Sign Language Recognition System V2.
=====================================================================

Usage:
------
1. Launch Modern Glassmorphism Webview GUI (Default):
   python run_app.py

2. Open in Default Web Browser (Chrome/Edge/Firefox):
   python run_app.py --browser

3. Launch Classic Tkinter Desktop GUI:
   python run_app.py --desktop

4. Launch Lightweight OpenCV HUD Mode:
   python run_app.py --opencv

5. Specify Camera Device Index:
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
        "--desktop",
        action="store_true",
        help="Launch the classic Tkinter desktop GUI instead of the modern webview interface.",
    )
    parser.add_argument(
        "--opencv",
        action="store_true",
        help="Launch the lightweight OpenCV HUD window instead of the webview interface.",
    )
    parser.add_argument(
        "--browser",
        action="store_true",
        help="Open modern web UI in default browser (Chrome/Edge) instead of PyWebView window.",
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
        from src.ui.opencv_app import run_opencv_app

        run_opencv_app(camera_index=args.camera)
    elif args.desktop:
        print("[INFO] Launching Classic Tkinter Desktop GUI...")
        from src.ui.desktop_app import SignLanguageApp

        app = SignLanguageApp()
        app.mainloop()
    else:
        print("[INFO] Launching Modern Glassmorphism Web Application...")
        from src.ui.web_app import run_web_app

        run_web_app(camera_index=args.camera, force_browser=args.browser)


if __name__ == "__main__":
    main()
