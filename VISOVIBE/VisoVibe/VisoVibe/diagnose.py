"""
Run this BEFORE app.py if the camera feed looks blank/gradient-only or the
vibration confidence seems random. It tells you:
  1. Which camera indexes actually give real (non-black) frames.
  2. Which audio input devices are available, so you can pick the right one.

Usage:
    python diagnose.py
"""

import sys
import numpy as np
import cv2
import sounddevice as sd

CAM_BACKEND = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY


def check_cameras(max_index=4):
    print("\n=== Camera check ===")
    found_good = False
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx, CAM_BACKEND)
        if not cap.isOpened():
            print(f"  index {idx}: could not open")
            cap.release()
            continue

        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            print(f"  index {idx}: opened but no frame returned")
            continue

        brightness = float(np.mean(frame))
        status = "OK (looks like real video)" if brightness > 5.0 else \
                 "SUSPICIOUS (near-black -- likely blocked/wrong backend)"
        found_good = found_good or brightness > 5.0
        print(f"  index {idx}: avg brightness={brightness:.1f}  ->  {status}")

    if not found_good:
        print(
            "\n  No working camera index found. Check:\n"
            "  - Windows Settings > Privacy & security > Camera > "
            "'Let desktop apps access your camera' is ON\n"
            "  - No other app (Teams/Zoom/Camera app/browser tab) currently "
            "has the camera open\n"
            "  - Try unplugging/replugging an external webcam if you have one"
        )
    else:
        print("\n  Set CAM_INDEX in app.py to whichever index showed OK above.")


def check_audio_devices():
    print("\n=== Audio input devices ===")
    devices = sd.query_devices()
    default_input = sd.default.device[0]
    for i, d in enumerate(devices):
        if d["max_input_channels"] > 0:
            marker = "  <-- current default" if i == default_input else ""
            print(f"  [{i}] {d['name']}  "
                  f"(inputs: {d['max_input_channels']}, "
                  f"samplerate: {d['default_samplerate']:.0f}){marker}")
    print(
        "\n  If the default isn't your laptop's built-in mic (e.g. it's "
        "picking a webcam mic array or virtual device), pass the correct "
        "index as `device=` when constructing ThroatVibrationSensor in "
        "app.py, e.g. ThroatVibrationSensor(device=2)."
    )


if __name__ == "__main__":
    check_cameras()
    check_audio_devices()
