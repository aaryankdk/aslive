"""
ASLive — Real-Time ASL Word Translation Backend
================================================
"""

import base64
import json
import os
import ssl
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
# macOS + python.org Python builds don't inherit the system's trusted root
# certificates, which makes urllib.request fail HTTPS downloads with
# CERTIFICATE_VERIFY_FAILED. If `certifi` is available, use its CA bundle
# explicitly so model downloads work regardless of whether the one-time
# "Install Certificates.command" script was run.
# ---------------------------------------------------------------------------
try:
    import certifi
    _SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
    _HTTPS_OPENER = urllib.request.build_opener(urllib.request.HTTPSHandler(context=_SSL_CONTEXT))
    urllib.request.install_opener(_HTTPS_OPENER)
except ImportError:
    print("[startup] certifi not installed — if model downloads fail with "
          "CERTIFICATE_VERIFY_FAILED, run `pip install certifi` or, on macOS, "
          "run the 'Install Certificates.command' that ships with your Python install.")

# ---------------------------------------------------------------------------
# Constants — must match preprocessing.py
# ---------------------------------------------------------------------------
MAX_FRAMES             = 30
FEATURE_SIZE           = 147
POSE_IDS               = [0, 11, 12, 13, 14, 15, 16]
# Two lightweight models replace the old heavy Holistic model (which also ran a
# full 478-point face mesh that was never used — see comment near model init).
POSE_MODEL_PATH        = "pose_landmarker_lite.task"
HAND_MODEL_PATH        = "hand_landmarker.task"
# Time-based cadence (independent of incoming frame rate):
#   PREDICT_INTERVAL_SECONDS - minimum time between two inference runs on the buffer
#   DISPLAY_GAP_SECONDS      - minimum time between two predictions accepted into
#                              the transcript, so results don't overlap/collide
PREDICT_INTERVAL_SECONDS = 2.0
DISPLAY_GAP_SECONDS      = 3.0

# ---------------------------------------------------------------------------
# WLASL-100 label map  (index → word)
#
# nslt_100.json only maps video_id -> {subset, action: [class_idx, start, end]}.
# It does NOT contain the English gloss/word itself, so it can't supply labels
# on its own. WLASL_v0_3.json is the full dataset manifest and DOES contain the
# gloss for every video_id (entry["gloss"] with entry["instances"][*]["video_id"]).
#
# We join the two files on video_id:
#   nslt_100.json   video_id -> class_idx   (which of the 100 classes)
#   WLASL_v0_3.json video_id -> gloss       (the actual English word)
# to build a complete, verified class_idx -> word list of length 100.
# ---------------------------------------------------------------------------
import os
import json as _json

_model_dir       = os.path.join(os.path.dirname(__file__), "model")
_nslt_path       = os.path.join(_model_dir, "nslt_100.json")

# Accept either naming convention for the full WLASL manifest file.
_wlasl_candidates = ["WLASL_v0_3.json", "WLASL_v0.3.json", "WLASL_v0-3.json"]
_wlasl_full_path = next(
    (p for p in (os.path.join(_model_dir, name) for name in _wlasl_candidates) if os.path.exists(p)),
    None,
)
if _wlasl_full_path is None:
    raise FileNotFoundError(
        f"Could not find the WLASL manifest file in {_model_dir}. "
        f"Expected one of: {_wlasl_candidates}"
    )

with open(_nslt_path) as _f:
    _nslt_data = _json.load(_f)

with open(_wlasl_full_path) as _f:
    _wlasl_full = _json.load(_f)

# 1. Build video_id -> gloss from the full WLASL manifest
_vid_to_gloss = {}
for _entry in _wlasl_full:
    _gloss = _entry["gloss"]
    for _inst in _entry["instances"]:
        _vid_to_gloss[_inst["video_id"]] = _gloss

# 2. Walk nslt_100.json, resolve each video_id's class_idx to its gloss
WLASL100_WORDS = [None] * 100
_conflicts = []
for _video_id, _meta in _nslt_data.items():
    _class_idx = _meta["action"][0]
    _gloss     = _vid_to_gloss.get(_video_id)

    if _gloss is None:
        continue  # video not found in WLASL_v0_3.json, skip
    if not (0 <= _class_idx < 100):
        continue

    if WLASL100_WORDS[_class_idx] is not None and WLASL100_WORDS[_class_idx] != _gloss:
        _conflicts.append((_class_idx, WLASL100_WORDS[_class_idx], _gloss))
    WLASL100_WORDS[_class_idx] = _gloss

_missing = [i for i, w in enumerate(WLASL100_WORDS) if w is None]
if _missing:
    raise RuntimeError(
        f"[label map] Could not resolve word(s) for class index(es): {_missing}. "
        "Check that nslt_100.json and WLASL_v0_3.json are in sync."
    )
if _conflicts:
    raise RuntimeError(f"[label map] Conflicting glosses found for class indices: {_conflicts}")

print("Successfully mapped words:", WLASL100_WORDS)
# ---------------------------------------------------------------------------
# Download MediaPipe task files if absent (lite pose model + hand model —
# no face mesh, since face landmarks were never used by this app).
# ---------------------------------------------------------------------------
_MODEL_DOWNLOADS = {
    POSE_MODEL_PATH: (
        "https://storage.googleapis.com/mediapipe-models/"
        "pose_landmarker/pose_landmarker_lite/float16/latest/"
        "pose_landmarker_lite.task"
    ),
    HAND_MODEL_PATH: (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/latest/"
        "hand_landmarker.task"
    ),
}
for _path, _url in _MODEL_DOWNLOADS.items():
    if not os.path.exists(_path):
        print(f"[startup] Downloading MediaPipe model to {_path} …")
        urllib.request.urlretrieve(_url, _path)
        print("[startup] Download complete.")

# ---------------------------------------------------------------------------
# MediaPipe Pose + Hand landmarkers
# VIDEO mode is used instead of IMAGE mode so that MediaPipe receives a
# monotonically-increasing timestamp on every call. This gives it the context
# it needs to apply the correct non-square projection matrix for the 640x480
# frame, eliminating the NORM_RECT / IMAGE_DIMENSIONS warning.
#
# Holistic (previous implementation) always computes pose + hands + a full
# face mesh in one call. This app never reads face landmarks, so running
# Holistic paid the full cost of face tracking on every frame for nothing —
# this was the main cause of the low real-world FPS. Using the dedicated
# lite Pose model + Hand model instead skips face tracking entirely and is
# significantly cheaper per frame.
# ---------------------------------------------------------------------------
_pose_opts = vision.PoseLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=POSE_MODEL_PATH),
    running_mode=vision.RunningMode.VIDEO,
    num_poses=1,
    output_segmentation_masks=False,
)
_hand_opts = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=HAND_MODEL_PATH),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
)
_pose_landmarker = vision.PoseLandmarker.create_from_options(_pose_opts)
_hand_landmarker = vision.HandLandmarker.create_from_options(_hand_opts)
_frame_timestamp_ms: int = 0  # incremented per frame


def extract_features(pose_result, hand_result):
    """147-D feature vector, identical to preprocessing.py."""
    pose = np.zeros((7, 3), dtype=np.float32)
    lh   = np.zeros((21, 3), dtype=np.float32)
    rh   = np.zeros((21, 3), dtype=np.float32)

    if pose_result.pose_landmarks:
        p    = pose_result.pose_landmarks[0]   # single-person: first detected pose
        pose = np.array([[p[i].x, p[i].y, p[i].z] for i in POSE_IDS], dtype=np.float32)
        pose -= (pose[1] + pose[2]) / 2.0

    for hand_lms, handedness in zip(hand_result.hand_landmarks, hand_result.handedness):
        label  = handedness[0].category_name  # "Left" or "Right"
        matrix = lh if label == "Left" else rh
        matrix[:] = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms], dtype=np.float32)
        matrix -= matrix[0]

    feat = np.vstack([pose, lh, rh]).flatten()
    return feat / (np.max(np.abs(feat)) or 1.0)


def landmarks_to_dict(pose_result, hand_result):
    """Serialise landmark coordinates for the frontend overlay canvas."""
    def lm_list(lms):
        if not lms:
            return []
        return [{"x": lm.x, "y": lm.y} for lm in lms]

    pose_pts = []
    if pose_result.pose_landmarks:
        p = pose_result.pose_landmarks[0]
        for i in POSE_IDS:
            pose_pts.append({"x": p[i].x, "y": p[i].y})

    left_hand, right_hand = [], []
    for hand_lms, handedness in zip(hand_result.hand_landmarks, hand_result.handedness):
        label = handedness[0].category_name
        if label == "Left":
            left_hand = lm_list(hand_lms)
        else:
            right_hand = lm_list(hand_lms)

    return {
        "pose":       pose_pts,
        "left_hand":  left_hand,
        "right_hand": right_hand,
    }


# ---------------------------------------------------------------------------
# Sliding window + state
# ---------------------------------------------------------------------------
_frame_buffer: deque      = deque(maxlen=MAX_FRAMES)
_last_predict_time: float = 0.0   # when inference was last run on the buffer
_last_word: str            = ""
_last_word_time: float     = 0.0  # when a prediction was last accepted for display/transcript
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
    global _last_predict_time, _last_word, _last_word_time, _cached_prediction
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

        _t0 = time.time()
        image_bytes = base64.b64decode(b64)
        pil_img = Image.open(BytesIO(image_bytes)).convert("RGB").resize((640, 480))
        frame   = np.array(pil_img, dtype=np.uint8)
        _t1 = time.time()

        global _frame_timestamp_ms
        _frame_timestamp_ms += 33   # matches the 33 ms capture interval (30 fps)
        mp_image     = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        _t2 = time.time()
        pose_result  = _pose_landmarker.detect_for_video(mp_image, _frame_timestamp_ms)
        _t3 = time.time()
        hand_result  = _hand_landmarker.detect_for_video(mp_image, _frame_timestamp_ms)
        _t4 = time.time()
        feat         = extract_features(pose_result, hand_result)
        landmarks    = landmarks_to_dict(pose_result, hand_result)
        _t5 = time.time()

        print(
            f"[TIMING] decode={1000*(_t1-_t0):.1f}ms  "
            f"mp_image={1000*(_t2-_t1):.1f}ms  "
            f"pose={1000*(_t3-_t2):.1f}ms  "
            f"hand={1000*(_t4-_t3):.1f}ms  "
            f"postproc={1000*(_t5-_t4):.1f}ms  "
            f"TOTAL={1000*(_t5-_t0):.1f}ms",
            flush=True,
        )
        hands_detected = bool(hand_result.hand_landmarks)

        _frame_buffer.append(feat)
        buffer_fill = len(_frame_buffer)

        # Run inference at most once every PREDICT_INTERVAL_SECONDS, regardless
        # of how fast frames are arriving from the frontend.
        if (
            buffer_fill == MAX_FRAMES
            and (now - _last_predict_time) >= PREDICT_INTERVAL_SECONDS
        ):
            _last_predict_time = now
            sequence = np.array(_frame_buffer, dtype=np.float32)
            word, confidence = predictor.predict(sequence)

            # Only accept this prediction into the transcript if enough time has
            # passed since the last accepted one, so consecutive outputs can't
            # collide/overlap on screen.
            now_predict = time.time()
            is_new = (now_predict - _last_word_time) >= DISPLAY_GAP_SECONDS
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
    global _last_predict_time, _last_word, _last_word_time, _cached_prediction
    global _last_frame_time, _fps_avg
    _frame_buffer.clear()
    _last_predict_time = 0.0
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
    # debug=False: the Werkzeug debugger/reloader adds real per-request overhead
    # and was contributing to the low FPS. threaded=True stays on so a slow
    # request doesn't fully block e.g. /api/reset.
    app.run(debug=False, port=5000, threaded=True)