import cv2
import numpy as np
import mediapipe as mp
import base64
from typing import List, Dict, Any

# --- Mediapipe 초기화 ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=2)

# --- 랜드마크 인덱스 ---
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_ANKLE = 27
RIGHT_ANKLE = 28
LEFT_FOOT_INDEX = 31
RIGHT_FOOT_INDEX = 32

# --- 핵심 서비스 함수 ---

def extract_landmarks_from_timestamps(video_path: str, timestamps: List[float]) -> List[Dict[str, Any]]:
    """타임스탬프에 해당하는 프레임의 랜드마크 데이터를 추출합니다."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file")

    step_data = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        raise ValueError("Could not determine video FPS.")

    for i, ts in enumerate(timestamps):
        frame_number = int(ts * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if not ret:
            continue

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_world_landmarks and results.pose_landmarks:
            step_data.append({
                "step": i + 1,
                "frame_number": frame_number,
                "world_landmarks": results.pose_world_landmarks,
                "image_landmarks": results.pose_landmarks
            })
    cap.release()
    return step_data

# Import analysis functions from analysis module
from analysis import analyze_torso_angle, analyze_foot_crossover_by_x

def analyze_sliding_stability(marked_steps: list) -> dict:
    """5스텝 안정성 분석 (현재는 빈 껍데기)."""
    return {5: []}  # Pydantic 모델에 맞게 빈 피드백 반환

# --- 시각화 로직 (최신 analysis.py 기준) ---

def visualize_analysis(video_path: str, marked_steps: list, torso_results: dict) -> Dict[str, str]:
    """Visualize torso analysis on video frames and encode as Base64.
    
    Draws torso vectors and colored landmarks on frames for steps 2-5.
    """
    annotated_images_b64 = {}
    angles = torso_results.get('angles', {})
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {}

    LANDMARK_COLORS = {
        LEFT_SHOULDER: (0, 0, 255),       # Red
        RIGHT_SHOULDER: (0, 165, 255),    # Orange
        LEFT_HIP: (0, 255, 255),          # Yellow
        RIGHT_HIP: (0, 255, 0),           # Green
    }

    for step_num in [2, 3, 4, 5]:
        try:
            step_info = next(item for item in marked_steps if item["step"] == step_num)
            frame_num = step_info['frame_number']
            image_landmarks = step_info['image_landmarks']
        except (StopIteration, KeyError):
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)
        ret, frame = cap.read()
        if not ret:
            continue
        
        h, w, _ = frame.shape

        for landmark_idx, color in LANDMARK_COLORS.items():
            landmark = image_landmarks.landmark[landmark_idx]
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(frame, (cx, cy), 10, color, -1)

        shoulder_l = image_landmarks.landmark[LEFT_SHOULDER]
        shoulder_r = image_landmarks.landmark[RIGHT_SHOULDER]
        hip_l = image_landmarks.landmark[LEFT_HIP]
        hip_r = image_landmarks.landmark[RIGHT_HIP]
        shoulder_mid_px = (int((shoulder_l.x + shoulder_r.x) * w / 2), int((shoulder_l.y + shoulder_r.y) * h / 2))
        hip_mid_px = (int((hip_l.x + hip_r.x) * w / 2), int((hip_l.y + hip_r.y) * h / 2))

        cv2.line(frame, hip_mid_px, shoulder_mid_px, (255, 255, 0), 3)
        
        angle = angles.get(step_num)
        if angle is not None:
            cv2.putText(frame, f"Step {step_num} Torso Tilt: {angle:.2f} deg", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        _, buffer = cv2.imencode('.jpg', frame)
        img_b64 = base64.b64encode(buffer).decode('utf-8')
        annotated_images_b64[f"Step_{step_num}_Analysis"] = img_b64

    cap.release()
    return annotated_images_b64
