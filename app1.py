"""
ASLive — Real-Time ASL Word Translation Backend
================================================
"""

import base64
import json
import os
import time
import urllib.request
from collections import deque
from io import BytesIO

import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image

from model.predictor import ASLPredictor

# ---------------------------------------------------------------------------
# Constants — must match preprocessing.py
# ---------------------------------------------------------------------------
MAX_FRAMES             = 30
FEATURE_SIZE           = 147
POSE_IDS               = [0, 11, 12, 13, 14, 15, 16]
MODEL_PATH             = "holistic_landmarker.task"
# CHANGED: Lowered stride from 30 to 2 for frequent window evaluations at 30 FPS
PREDICT_EVERY_N_FRAMES = 2
DEDUP_COOLDOWN_SECONDS = 1.5

# ---------------------------------------------------------------------------
# WLASL-100 label map  (index → word)
# Loaded from model/nslt_100.json if present; falls back to numeric strings.
# nslt_100.json maps video_id → {action: [class_idx, ...], ...}
# We derive the word list from the WLASL-100 canonical ordering.
# ---------------------------------------------------------------------------
import os
import json as _json

# 1. Load the raw NSLT json file
json_path = os.path.join(os.path.dirname(__file__), "model", "nslt_100.json")
with open(json_path) as _f:
    nslt_data = _json.load(_f)

# 2. Initialize a fixed list of size 100
WLASL100_WORDS = [None] * 100

# 3. Populate it by parsing the internal structure
for video_id, meta in nslt_data.items():
    # In standard WLASL nslt configs, 'action' contains [class_id, start_frame, end_frame]
    # and 'text' contains the target English word.
    class_idx = meta["action"][0] 
    word = meta["text"]
    
    # Safety check: make sure it fits within your 100-word index mapping
    if 0 <= class_idx < 100:
        WLASL100_WORDS[class_idx] = word

print("Successfully mapped words:", WLASL100_WORDS)
# ---------------------------------------------------------------------------
# Download MediaPipe holistic task file if absent
# ---------------------------------------------------------------------------
if not os.path.exists(MODEL_PATH):
    _URL = (
        "https://storage.googleapis.com/mediapipe-models/"
        "holistic_landmarker/holistic_landmarker/float16/latest/"
        "holistic_landmarker.task"
    )
    print(f"[startup] Downloading MediaPipe model to {MODEL_PATH} …")
    urllib.request.urlretrieve(_URL, MODEL_PATH)
    print("[startup] Download complete.")

# ---------------------------------------------------------------------------
# MediaPipe Holistic landmarker
# VIDEO mode is used instead of IMAGE mode so that MediaPipe receives a
# monotonically-increasing timestamp on every call. This gives it the context
# it needs to apply the correct non-square projection matrix for the 640x480
# frame, eliminating the NORM_RECT / IMAGE_DIMENSIONS warning.
# ---------------------------------------------------------------------------
_mp_opts = vision.HolisticLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.VIDEO,
    output_segmentation_mask=False,
)
_landmarker = vision.HolisticLandmarker.create_from_options(_mp_opts)
_frame_timestamp_ms: int = 0  # incremented per frame


def extract_features(results):
    """147-D feature vector, identical to preprocessing.py."""
    pose = np.zeros((7, 3), dtype=np.float32)
    lh   = np.zeros((21, 3), dtype=np.float32)
    rh   = np.zeros((21, 3), dtype=np.float32)

    if results.pose_landmarks:
        p    = results.pose_landmarks
        pose = np.array([[p[i].x, p[i].y, p[i].z] for i in POSE_IDS], dtype=np.float32)
        pose -= (pose[1] + pose[2]) / 2.0

    for hand_res, matrix in [
        (results.left_hand_landmarks,  lh),
        (results.right_hand_landmarks, rh),
    ]:
        if hand_res:
            matrix[:] = np.array([[lm.x, lm.y, lm.z] for lm in hand_res], dtype=np.float32)
            matrix -= matrix[0]

    feat = np.vstack([pose, lh, rh]).flatten()
    return feat / (np.max(np.abs(feat)) or 1.0)


def landmarks_to_dict(results):
    """Serialise landmark coordinates for the frontend overlay canvas."""
    def lm_list(lms):
        if not lms:
            return []
        return [{"x": lm.x, "y": lm.y} for lm in lms]

    pose_pts = []
    if results.pose_landmarks:
        for i in POSE_IDS:
            lm = results.pose_landmarks[i]
            pose_pts.append({"x": lm.x, "y": lm.y})

    return {
        "pose":       pose_pts,
        "left_hand":  lm_list(results.left_hand_landmarks),
        "right_hand": lm_list(results.right_hand_landmarks),
    }


# ---------------------------------------------------------------------------
# Sliding window + state
# ---------------------------------------------------------------------------
_frame_buffer: deque      = deque(maxlen=MAX_FRAMES)
_frames_since_last_predict: int = 0
_last_word: str            = ""
_last_word_time: float     = 0.0
_cached_prediction: dict   = {"prediction": None, "confidence": 0.0, "is_new": False}

# FPS Metrics Trackers
_last_frame_time: float    = 0.0
_fps_avg: float            = 0.0

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)

predictor = ASLPredictor(
    model_path=os.path.join(os.path.dirname(__file__), "model", "converted_model.tflite"),
    label_map=WLASL100_WORDS,
)


# ===========================================================================
# ROUTES
# ===========================================================================

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/translate")
def translate():
    return render_template("translate.html")


# ===========================================================================
# API — /api/predict
# ===========================================================================

@app.route("/api/predict", methods=["POST"])
def predict():
    global _frames_since_last_predict, _last_word, _last_word_time, _cached_prediction
    global _last_frame_time, _fps_avg

    try:
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"error": "No image provided"}), 400

        # ── Calculate Real-Time Video FPS ───────────────────────────────────
        now = time.time()
        if _last_frame_time > 0:
            time_delta = now - _last_frame_time
            if time_delta > 0:
                current_fps = 1.0 / time_delta
                # Exponential moving average smooths frame intervals out nicely
                _fps_avg = (0.9 * _fps_avg) + (0.1 * current_fps) if _fps_avg > 0 else current_fps
        _last_frame_time = now

        print(f"[INFO] Video Input Speed: {_fps_avg:.1f} FPS", flush=True)
        # ────────────────────────────────────────────────────────────────────

        b64 = data["image"]
        if "," in b64:
            b64 = b64.split(",")[1]
        image_bytes = base64.b64decode(b64)

        pil_img = Image.open(BytesIO(image_bytes)).convert("RGB").resize((640, 480))
        frame   = np.array(pil_img, dtype=np.uint8)

        global _frame_timestamp_ms
        # CHANGED: Updated comment text to correctly state 30 FPS representation
        _frame_timestamp_ms += 33   # matches the 33 ms capture interval (30 fps)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        results  = _landmarker.detect_for_video(mp_image, _frame_timestamp_ms)
        feat     = extract_features(results)
        landmarks = landmarks_to_dict(results)
        hands_detected = bool(results.left_hand_landmarks or results.right_hand_landmarks)

        _frame_buffer.append(feat)
        _frames_since_last_predict += 1
        buffer_fill = len(_frame_buffer)

        if (
            buffer_fill == MAX_FRAMES
            and _frames_since_last_predict >= PREDICT_EVERY_N_FRAMES
        ):
            _frames_since_last_predict = 0
            sequence = np.array(_frame_buffer, dtype=np.float32)
            word, confidence = predictor.predict(sequence)

            now_predict = time.time()
            is_new = not (
                word == _last_word
                and (now_predict - _last_word_time) < DEDUP_COOLDOWN_SECONDS
            )
            if is_new:
                _last_word      = word
                _last_word_time = now_predict

            _cached_prediction = {
                "prediction":  word,
                "confidence":  confidence,
                "is_new":      is_new,
            }

        return jsonify({
            **_cached_prediction,
            "buffer_fill":   buffer_fill,
            "buffer_max":    MAX_FRAMES,
            "landmarks":     landmarks,
            "hands_detected": hands_detected,
        })

    except Exception as e:
        print(f"[predict] Error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/reset", methods=["POST"])
def reset():
    global _frames_since_last_predict, _last_word, _last_word_time, _cached_prediction
    global _last_frame_time, _fps_avg
    _frame_buffer.clear()
    _frames_since_last_predict = 0
    _last_word      = ""
    _last_word_time = 0.0
    _last_frame_time = 0.0
    _fps_avg         = 0.0
    _cached_prediction = {"prediction": None, "confidence": 0.0, "is_new": False}
    return jsonify({"status": "buffer and FPS cleared"})


@app.route("/api/model-info", methods=["GET"])
def model_info():
    return jsonify(predictor.get_model_info())


@app.route("/api/text-to-asl", methods=["POST"])
def text_to_asl():
    try:
        data = request.get_json()
        if not data or "text" not in data:
            return jsonify({"error": "No text provided"}), 400
        signs = []
        for char in data["text"].upper():
            if char.isalpha():
                signs.append({"letter": char, "image_url": f"/static/images/sign_{char.lower()}.png"})
            elif char == " ":
                signs.append({"letter": " ", "image_url": None})
        return jsonify({"signs": signs})
    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__ == "__main__":
    # Add threaded=True
    app.run(debug=True, port=5000, threaded=True)