import os
import json
import cv2
import numpy as np
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

JSON_PATH, VIDEO_DIR, OUT_DIR, MODEL_PATH = "nslt_100.json", "videos", "processed_data", "holistic_landmarker.task"
MAX_FRAMES, FEATURE_SIZE, POSE_IDS = 30, 147, [0, 11, 12, 13, 14, 15, 16]

if not os.path.exists(MODEL_PATH):
    url = "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task"
    urllib.request.urlretrieve(url, MODEL_PATH)

def extract_features(results):
    pose, lh, rh = np.zeros((7, 3)), np.zeros((21, 3)), np.zeros((21, 3))
    if results.pose_landmarks:
        p_list = results.pose_landmarks
        pose = np.array([[p_list[i].x, p_list[i].y, p_list[i].z] for i in POSE_IDS])
        pose -= (pose[1] + pose[2]) / 2.0
    for hand_res, matrix in [(results.left_hand_landmarks, lh), (results.right_hand_landmarks, rh)]:
        if hand_res:
            matrix[:] = [[lm.x, lm.y, lm.z] for lm in hand_res]
            matrix -= matrix[0]
    feat = np.vstack([pose, lh, rh])
    return feat.flatten() / (np.max(np.abs(feat)) or 1.0)

def run_pipeline():
    if not os.path.exists(JSON_PATH):
        return
    with open(JSON_PATH) as f:
        data = json.load(f)
    tasks = [(os.path.join(VIDEO_DIR, f"{k}.mp4"), v['action'][0], v['subset'])
             for k, v in data.items() if os.path.exists(os.path.join(VIDEO_DIR, f"{k}.mp4"))]
    splits = {s: {'X': [], 'y': []} for s in ['train', 'val', 'test']}
    opts = vision.HolisticLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.IMAGE,
        output_segmentation_mask=False
    )
    with vision.HolisticLandmarker.create_from_options(opts) as landmarker:
        for idx, (path, label, split) in enumerate(tasks, 1):
            cap, seq = cv2.VideoCapture(path), []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (640, 480))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                try:
                    res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=frame))
                    seq.append(extract_features(res))
                except Exception as e:
                    print(f"Frame error: {e}", flush=True)
            cap.release()
            if not seq:
                continue
            if len(seq) >= MAX_FRAMES:
                seq = [seq[i] for i in np.linspace(0, len(seq) - 1, MAX_FRAMES, dtype=int)]
            else:
                seq += [np.zeros(FEATURE_SIZE)] * (MAX_FRAMES - len(seq))
            if split in splits:
                splits[split]['X'].append(seq)
                splits[split]['y'].append(label)
            print(f"Video {idx}/{len(tasks)}", flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    for name, d in splits.items():
        X, y = np.array(d['X'], dtype=np.float32), np.array(d['y'], dtype=np.int64)
        if X.ndim == 1 or X.shape[0] == 0:
            X = X.reshape(0, MAX_FRAMES, FEATURE_SIZE)
        np.save(os.path.join(OUT_DIR, f"X_{name}.npy"), X)
        np.save(os.path.join(OUT_DIR, f"y_{name}.npy"), y)

if __name__ == "__main__":
    run_pipeline()