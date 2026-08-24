
import sys
import time
import threading

import cv2
import numpy as np
from flask import Flask, Response, render_template, jsonify

from depth_mapper import DepthMapper
from vibration_sensor import ThroatVibrationSensor
from lip_reader import LipReader
from fusion import fuse_prediction

app = Flask(__name__)

CAM_INDEX = 0
DEPTH_FRAME_SKIP = 5  # run the (expensive) depth model every Nth frame
STREAM_INTERVAL = 0.05  # ~20fps cap on the MJPEG streams themselves

# Windows' default OpenCV camera backend (MSMF) frequently returns
# black/empty frames while still reporting isOpened()==True.
CAM_BACKEND = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY

print("Loading lip-reading model (MediaPipe + Keras)...")
lip_reader = LipReader()

print("Loading MiDaS_small depth model (first run downloads weights)...")
depth_mapper = DepthMapper(model_type="MiDaS_small")

vibration_sensor = ThroatVibrationSensor()

_state_lock = threading.Lock()
_state = {
    "camera_frame": None,   # BGR, with lip-landmark overlay drawn on
    "depth_frame": None,    # BGR, colorized depth map
    "lip_result": None,     # dict from LipReader.process()
    "proximity": 0.0,
}


def capture_loop():
    """
    The ONE place that owns the webcam. Both the lip-reading model and the
    depth model consume frames from here -- this replaces the three
    separate camera opens (depth module, lip predict.py, browser
    getUserMedia) that couldn't run at the same time.
    """
    cap = cv2.VideoCapture(CAM_INDEX, CAM_BACKEND)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open webcam at index {CAM_INDEX}. "
            "Run diagnose.py to check indexes/permissions."
        )

    frame_count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue

        frame = cv2.flip(frame, 1)  # mirror, matches how you look at the screen
        frame_count += 1

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        lip_result = lip_reader.process(rgb)

        annotated = frame.copy()
        if lip_result["face_points"]:
            for (x, y) in lip_result["face_points"]:
                cv2.circle(annotated, (x, y), 2, (0, 255, 0), -1)
        cv2.rectangle(annotated, (10, 10), (340, 65), (0, 0, 0), -1)
        cv2.putText(annotated, lip_result["word"], (20, 48), 0, 1.0, (0, 255, 0), 2)

        if frame_count % DEPTH_FRAME_SKIP == 0:
            depth_map = depth_mapper.predict(frame)
            colored_depth = depth_mapper.colorize(depth_map)
            proximity = depth_mapper.center_proximity(depth_map)
            with _state_lock:
                _state["depth_frame"] = colored_depth
                _state["proximity"] = proximity

        with _state_lock:
            _state["camera_frame"] = annotated
            _state["lip_result"] = lip_result


def _mjpeg_stream(frame_key):
    while True:
        with _state_lock:
            frame = _state.get(frame_key)
        if frame is not None:
            ok, buffer = cv2.imencode(".jpg", frame)
            if ok:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
        time.sleep(STREAM_INTERVAL)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/camera_feed")
def camera_feed():
    return Response(_mjpeg_stream("camera_frame"),
                     mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/depth_feed")
def depth_feed():
    return Response(_mjpeg_stream("depth_frame"),
                     mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/signals")
def signals():
    vib = vibration_sensor.read()
    with _state_lock:
        lip_result = _state["lip_result"] or {
            "word": "...", "confidence": 0.0, "top2_word": "", "top2_confidence": 0.0
        }
        proximity = _state["proximity"]

    fused = fuse_prediction(lip_result, vib["confidence"])

    return jsonify({
        "word": fused["word"],
        "confidence": fused["confidence"],
        "vibration_confidence": vib["confidence"],
        "obstacle_proximity": proximity,
        "obstacle_alert": proximity > 0.75,
    })


if __name__ == "__main__":
    vibration_sensor.start()
    capture_thread = threading.Thread(target=capture_loop, daemon=True)
    capture_thread.start()
    try:
        app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
    finally:
        vibration_sensor.stop()
