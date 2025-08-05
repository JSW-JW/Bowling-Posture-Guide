import cv2
import numpy as np
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
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

# --- 분석 로직 (최신 analysis.py 기준) ---

def get_landmark_point(landmarks: landmark_pb2.NormalizedLandmarkList, index: int) -> np.ndarray:
    return np.array([landmarks.landmark[index].x, landmarks.landmark[index].y])

def calculate_torso_tilt_from_image(landmarks) -> float:
    """오른쪽으로 기울수록 양수(+) 값을 반환하는 각도 계산"""
    shoulder_l = get_landmark_point(landmarks, LEFT_SHOULDER)
    shoulder_r = get_landmark_point(landmarks, RIGHT_SHOULDER)
    hip_l = get_landmark_point(landmarks, LEFT_HIP)
    hip_r = get_landmark_point(landmarks, RIGHT_HIP)
    shoulder_mid_pt = (shoulder_l + shoulder_r) / 2
    hip_mid_pt = (hip_l + hip_r) / 2
    torso_vector = shoulder_mid_pt - hip_mid_pt
    angle = np.degrees(np.arctan2(torso_vector[0], -torso_vector[1]))
    return angle

def analyze_torso_angle(marked_steps: list) -> dict:
    feedback = {2: [], 3: [], 4: [], 5: []}
    angles = {}
    try:
        landmarks = {s['step']: s['image_landmarks'] for s in marked_steps}
    except (StopIteration, KeyError):
        return {"feedback": {s: ["Torso analysis error"] for s in feedback.keys()}, "angles": angles}

    angle_step2 = calculate_torso_tilt_from_image(landmarks[2])
    angle_step3 = calculate_torso_tilt_from_image(landmarks[3])
    angle_step4 = calculate_torso_tilt_from_image(landmarks[4])
    angle_step5 = calculate_torso_tilt_from_image(landmarks[5])
    angles = {2: angle_step2, 3: angle_step3, 4: angle_step4, 5: angle_step5}

    if angle_step3 > angle_step2:
        feedback[3].append("[Torso] Good: Proper tilt.")
    else:
        feedback[3].append("[Torso] Advice: Tilt more.")
    if angle_step4 > angle_step3 + 5:
        feedback[4].append("[Torso] Good: Proper tilt.")
    else:
        feedback[4].append("[Torso] Advice: Tilt more.")
    if abs(angle_step5 - angle_step4) < 10:
        feedback[5].append("[Torso] Good: Angle maintained.")
    else:
        feedback[5].append("[Torso] Advice: Maintain angle.")
    return {"feedback": feedback, "angles": angles}

def analyze_foot_crossover_by_x(marked_steps: list) -> dict:
    feedback = {2: [], 3: [], 4: []}
    try:
        landmarks_dict = {s['step']: s['image_landmarks'].landmark for s in marked_steps}
    except (StopIteration, KeyError):
        return {s: ["Foot analysis error"] for s in feedback.keys()}

    for step_num in [2, 4]:
        lm = landmarks_dict[step_num]
        left_foot_x = lm[LEFT_FOOT_INDEX].x
        right_ankle_x = lm[RIGHT_ANKLE].x
        if left_foot_x + 0.08 < right_ankle_x:
             feedback[step_num].append(f"[Foot] Advice: Crossover needed more.")
        else:
             feedback[step_num].append(f"[Foot] Good: Crossover is sufficient.")

    lm3 = landmarks_dict[3]
    left_ankle_x_s3 = lm3[LEFT_ANKLE].x
    right_ankle_x_s3 = lm3[RIGHT_ANKLE].x
    if right_ankle_x_s3 > left_ankle_x_s3:
        feedback[3].append("[Foot] Good: Feet are uncrossed correctly.")
    else:
        feedback[3].append("[Foot] Advice: Feet should not be crossed.")
    return feedback

# --- 시각화 로직 (최신 analysis.py 기준) ---

def visualize_analysis(video_path: str, marked_steps: list, torso_results: dict) -> Dict[str, str]:
    """스텝 2,3,4,5 프레임 위에 몸통 벡터와 색상별 랜드마크를 그리고 Base64로 인코딩합니다."""
    annotated_images_b64 = {}
    angles = torso_results.get('angles', {})
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return {}

    LANDMARK_COLORS = {
        LEFT_SHOULDER: (0, 0, 255),       # 빨강
        RIGHT_SHOULDER: (0, 165, 255),    # 주황
        LEFT_HIP: (0, 255, 255),          # 노랑
        RIGHT_HIP: (0, 255, 0),           # 초록
    }

    for step_num in [2, 3, 4, 5]:
        try:
            step_info = next(item for item in marked_steps if item["step"] == step_num)
            frame_num = step_info['frame_number']
            image_landmarks = step_info['image_landmarks']
        except (StopIteration, KeyError): continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)
        ret, frame = cap.read()
        if not ret: continue
        
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