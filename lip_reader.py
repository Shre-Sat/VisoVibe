import time
from collections import deque

import numpy as np
import mediapipe as mp
import tensorflow as tf

# Same landmark indices and model architecture assumptions as the original
# collect_data.py / predict.py / train.py scripts -- kept identical so the
# already-trained model.keras / labels.npy files work without retraining.
MOUTH_POINTS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267,
                0, 37, 39, 40, 185, 80, 81, 82, 13, 312, 311, 310, 415, 308]

SEQUENCE_LENGTH = 30
CONFIDENCE_THRESHOLD = 0.70
VOTE_HISTORY = 8


class LipReader:
    """
    Wraps the trained lip-reading model (MediaPipe FaceLandmarker landmarks
    -> Bi-LSTM classifier) so it can run inside ONE shared webcam capture
    loop, instead of opening its own cv2.imshow() window and its own
    VideoCapture like the original predict.py did. Only one process can
    hold the webcam at a time, so this had to change for integration.

    Needs these files present (copy them from wherever they were trained):
        models/face_landmarker.task
        models/lip_model.keras
        models/labels.npy
    """

    def __init__(self, model_path="models/lip_model.keras",
                 labels_path="models/labels.npy",
                 landmarker_task_path="models/face_landmarker.task"):
        self.model = tf.keras.models.load_model(model_path)
        self.labels = np.load(labels_path)

        base_options = mp.tasks.BaseOptions(model_asset_path=landmarker_task_path)
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=1,
        )
        self.landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)

        self._sequence = deque(maxlen=SEQUENCE_LENGTH)
        self._vote_history = deque(maxlen=VOTE_HISTORY)
        self._last_word = "..."

    @staticmethod
    def _extract(face_landmarks):
        p = np.array([[face_landmarks[i].x, face_landmarks[i].y, face_landmarks[i].z]
                       for i in MOUTH_POINTS], dtype=np.float32)
        p[:, :2] -= np.mean(p[:, :2], axis=0)
        return p.flatten()

    def process(self, frame_rgb):
        """
        Call once per video frame (RGB, uint8 HxWx3 -- same convention the
        original scripts used after cv2.cvtColor(frame, COLOR_BGR2RGB)).

        Returns a dict:
            word              -- smoothed/majority-voted top prediction
            confidence        -- this frame's top-1 softmax value
            top2_word         -- runner-up word, for disambiguation
            top2_confidence   -- runner-up softmax value
            face_points       -- [(x,y), ...] pixel coords for overlay
                                 drawing, or None if no face detected
        """
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self.landmarker.detect_for_video(img, int(time.time() * 1000))

        top_conf = 0.0
        top2_word, top2_conf = "", 0.0
        face_points = None

        if result.face_landmarks:
            face = result.face_landmarks[0]
            self._sequence.append(self._extract(face))
            h, w = frame_rgb.shape[:2]
            face_points = [(int(face[i].x * w), int(face[i].y * h)) for i in MOUTH_POINTS]

        if len(self._sequence) == SEQUENCE_LENGTH:
            pred = self.model.predict(
                np.expand_dims(np.asarray(self._sequence), 0), verbose=0
            )[0]
            order = np.argsort(pred)[::-1]
            top_idx = order[0]
            second_idx = order[1] if len(order) > 1 else order[0]

            top_conf = float(pred[top_idx])
            top2_word = str(self.labels[second_idx])
            top2_conf = float(pred[second_idx])

            if top_conf > CONFIDENCE_THRESHOLD:
                self._vote_history.append(str(self.labels[top_idx]))
                vals, counts = np.unique(self._vote_history, return_counts=True)
                self._last_word = str(vals[np.argmax(counts)])

        return {
            "word": self._last_word,
            "confidence": top_conf,
            "top2_word": top2_word,
            "top2_confidence": top2_conf,
            "face_points": face_points,
        }
