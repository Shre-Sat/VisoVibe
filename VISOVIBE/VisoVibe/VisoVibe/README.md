# VocalSense AI — Unified System

This merges all three teammates' work into one Flask app: lip reading,
throat-vibration signal, and depth mapping/obstacle alert, driven by ONE
shared webcam capture loop and displayed on ONE UI.

## What changed from the separate pieces

- **Lip reading's `predict.py`** opened its own webcam and its own
  `cv2.imshow()` window. That's now `lip_reader.py` — same model, same
  landmark extraction, same voting logic, but it's a class with a
  `process(frame)` method that gets called from a shared loop instead of
  owning the camera itself.
- **The depth/vibration module's** own `cv2.VideoCapture` is gone — one
  camera can only really be held by one process reliably, so `depth_mapper.py`
  and `vibration_sensor.py` are unchanged, but `app.py` now feeds them
  frames from the single shared capture loop instead of each opening the
  camera separately.
- **The frontend's `getUserMedia()` camera and `Math.random()` prediction**
  are gone. The camera feed now streams from the backend (so it can include
  the lip-landmark overlay), and the prediction/confidence/vibration/depth
  values come from real `/signals` polling.
- **`fusion.py`** now has the actual weighting algorithm that was missing —
  see below.

## The weighting algorithm (what was missing)

`fusion.fuse_prediction()`:
1. Takes the lip model's top-1 and top-2 predictions + confidences.
2. If the top-2 words are both in `CONFUSABLE_PAIRS` (a set you fill in —
   see comments in `fusion.py`) AND their confidences are close (within
   `AMBIGUITY_MARGIN = 0.15`), vibration confidence picks between them via
   `resolve_ambiguous()`.
3. Otherwise the lip model's top-1 guess is trusted directly.
4. The **displayed confidence** is always a weighted blend:
   `0.8 * lip_confidence + 0.2 * vibration_confidence` — lip is primary,
   vibration is supporting. Tune `lip_weight` in `combined_confidence()`
   once you've watched both signals live.

`CONFUSABLE_PAIRS` starts empty because your current 8-word vocab
(hello/yes/no/thanks/stop/help/good/bye) wasn't built around visually
identical mouth shapes the way the original brief's p/b/m example was.
Watch the demo for a bit — if the model keeps swapping two specific words,
add that pair and vibration will help break the tie.

## Setup

```bash
cd vocalsense
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**You must copy your trained lip-reading files into `models/`:**
```
models/face_landmarker.task
models/lip_model.keras
models/labels.npy
```
These are the files `train.py` produced (`lip_model.keras`, `labels.npy`)
plus the MediaPipe task file `collect_data.py`/`predict.py` already
referenced. `app.py` won't start without them.

## Run

```bash
python diagnose.py   # confirm camera + mic are working, if you haven't already
python app.py
```

Open **http://localhost:5000**. First run downloads MiDaS depth weights
(needs internet once). You should see:
- Live camera feed with green mouth-landmark dots and the current
  predicted word overlaid.
- The "Speech Prediction" card updating with the fused word + confidence.
- The vibration bar reacting to humming/whispering near the mic.
- The depth thumbnail updating, with a red "Object close ahead" alert when
  something's centered and close.

## Performance note

Running MediaPipe + a Keras LSTM + MiDaS on CPU, in one process, every
frame, is heavier than any of the three running alone. If it's laggy:
- `DEPTH_FRAME_SKIP` in `app.py` (default 5) — raise it so depth runs less
  often; lip reading still runs every frame since word latency matters more.
- If it's still too slow, consider running lip reading every 2nd frame too
  (would need a small change to `capture_loop()` — ask if you want this
  wired in).

## Known rough edges / what to test before demo

- `CONFUSABLE_PAIRS` is empty — decide before demo day whether your vocab
  actually needs the disambiguation feature, or if it's fine to skip and
  just present vibration as a supporting confidence signal (still real,
  still worth demoing, just not doing tie-breaking).
- `rms_gate` and `low_band_hz` in `vibration_sensor.py` need tuning to
  whichever room/mic you're demoing on — do this in the room if you can,
  ambient noise varies a lot.
- The obstacle threshold `proximity > 0.75` in `app.py` needs the same
  live tuning.
