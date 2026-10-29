"""
Nitriva ML Launcher Script
Run this script to start the Nitriva Assistive Eye-Tracking System:
    python run.py
"""

import os
import sys
import time
import webbrowser
import threading
import uvicorn


def check_prerequisites():
    """Verify model weights and dependencies."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "models", "face_landmarker.task")

    print("=" * 65)
    print("           NITRIVA ASSISTIVE EYE-TRACKING SYSTEM")
    print("=" * 65)

    if not os.path.exists(model_path):
        print(f"[!] Warning: Model file missing at {model_path}")
        print("    Downloading MediaPipe FaceLandmarker model...")
        import urllib.request
        os.makedirs(os.path.join(base_dir, "models"), exist_ok=True)
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        urllib.request.urlretrieve(url, model_path)
        print("    Model downloaded successfully!")

    print(f"[*] ML Model Asset: Verified ({os.path.getsize(model_path):,} bytes)")
    print("[*] Python Runtime: " + sys.version.split()[0])
    print("[*] Local Web App: http://localhost:8000")
    print("=" * 65)


def open_browser():
    """Open default browser after a brief delay for server spin-up."""
    time.sleep(1.2)
    webbrowser.open("http://localhost:8000")


if __name__ == "__main__":
    check_prerequisites()

    # Open browser automatically in a background thread
    threading.Thread(target=open_browser, daemon=True).start()

    # Run FastAPI server
    uvicorn.run("src.server:app", host="0.0.0.0", port=8000, log_level="info")
