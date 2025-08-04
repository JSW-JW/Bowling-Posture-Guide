import cv2
import numpy as np
from mediapipe.framework.formats import landmark_pb2
import mediapipe as mp

# --- 랜드마크 인덱스 정의 ---
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_ANKLE = 27
RIGHT_ANKLE = 28
LEFT_FOOT_INDEX = 31
RIGHT_FOOT_INDEX = 32

def get_landmark_point(landmarks: landmark_pb2.NormalizedLandmarkList, index: int) -> np.ndarray:
    """랜드마크 리스트에서 특정 인덱스의 2D 랜드마크 좌표(x, y)를 numpy 배열로 반환합니다."""
    return np.array([landmarks.landmark[index].x, landmarks.landmark[index].y])

def calculate_torso_tilt_from_image(landmarks) -> float:
    """상체의 좌우 기울기를 계산합니다. (이미지 좌표 기준, 오른쪽 기울수록 양수)"""
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
    """상체 각도를 분석하고 스텝별 피드백과 각도 값을 반환합니다."""
    feedback = {3: [], 4: [], 5: []}
    angles = {}
    try:
        landmarks_step2 = next(item['image_landmarks'] for item in marked_steps if item["step"] == 2)
        landmarks_step3 = next(item['image_landmarks'] for item in marked_steps if item["step"] == 3)
        landmarks_step4 = next(item['image_landmarks'] for item in marked_steps if item["step"] == 4)
        landmarks_step5 = next(item['image_landmarks'] for item in marked_steps if item["step"] == 5)
    except (StopIteration, KeyError):
        return {"feedback": {3:["Torso analysis error"], 4:["Torso analysis error"], 5:["Torso analysis error"]}, "angles": angles}

    angle_step2 = calculate_torso_tilt_from_image(landmarks_step2)
    angle_step3 = calculate_torso_tilt_from_image(landmarks_step3)
    angle_step4 = calculate_torso_tilt_from_image(landmarks_step4)
    angle_step5 = calculate_torso_tilt_from_image(landmarks_step5)
    angles = {2: angle_step2, 3: angle_step3, 4: angle_step4, 5: angle_step5}

    if angle_step4 > angle_step3 + 5:
        feedback[4].append("[Torso] Good: Proper tilt.")
    else:
        feedback[4].append("[Torso] Advice: Tilt more.")
    if abs(angle_step5 - angle_step4) < 5:
        feedback[5].append("[Torso] Good: Angle maintained.")
    else:
        feedback[5].append("[Torso] Advice: Maintain angle.")
    return {"feedback": feedback, "angles": angles}

# 랜드마크 인덱스 정의
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28

import numpy as np

# 랜드마크 인덱스 정의
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_ANKLE = 27
RIGHT_ANKLE = 28

def analyze_foot_crossover_by_x(marked_steps: list) -> dict:
    """
    x좌표를 기준으로 Crossover를 분석합니다.
    안정적인 '골반 너비'를 기준으로 교차 비율을 계산합니다.
    """
    
    # --- 스텝별 조절 가능한 임계값 (Thresholds) ---
    # 기준이 골반 너비로 변경되었으므로, 이 값들은 다시 테스트하며 조절해야 할 수 있습니다.
    THRESHOLDS = {
        # 2스텝 기준: 가벼운 교차
        2: { "MIN_CROSSOVER_RATIO": 0.075, "MAX_CROSSOVER_RATIO": 0.25 }, # 오른발이 왼발보다 골반 너비의 15% 이상 왼쪽으로 가야 함
        # 4스텝 기준: 깊은 교차
        4: { "MIN_CROSSOVER_RATIO": 0.10, "MAX_CROSSOVER_RATIO": 0.35 }  # 오른발이 왼발보다 골반 너비의 30% 이상 왼쪽으로 가야 함
    }
    # -----------------------------------------

    feedback = {2: [], 3: [], 4: []}
    try:
        landmarks_dict = {s['step']: s['image_landmarks'].landmark for s in marked_steps if s['step'] in [2, 3, 4]}
        if not all(k in landmarks_dict for k in [2, 3, 4]):
            raise KeyError("One or more required steps are missing.")
    except (StopIteration, KeyError) as e:
        return {s: [f"Foot analysis error: {e}"] for s in feedback.keys()}

    # --- 2스텝 & 4스텝 (교차) 로직 ---
    for step_num in [2, 4]:
        lm = landmarks_dict[step_num]
        min_crossover_threshold = THRESHOLDS[step_num]["MIN_CROSSOVER_RATIO"]
        max_crossover_threshold = THRESHOLDS[step_num]["MAX_CROSSOVER_RATIO"]
        
        left_foot_x = lm[LEFT_FOOT_INDEX].x
        left_ankle_x = lm[LEFT_ANKLE].x
        right_ankle_x = lm[RIGHT_ANKLE].x
        
        # 정규화 기준을 '골반 너비'로 변경
        hip_width = abs(lm[LEFT_HIP].x - lm[RIGHT_HIP].x)
        
        # crossover_distance = right_ankle_x - left_ankle_x
        # crossover_distance = right_ankle_x - left_ankle_x
        # crossover_ratio = (crossover_distance / hip_width) if hip_width > 0 else 0
        
        # if min_crossover_threshold <= crossover_ratio <= max_crossover_threshold:
        if left_foot_x + 0.08 < right_ankle_x:
            feedback[step_num].append(f"[Foot] Advice: {step_num}, left_foot_x: {left_foot_x} right_ankle_x: {right_ankle_x}")
        else:
            feedback[step_num].append(f"[Foot] Good: Step {step_num} crossover is sufficient, left_foot_x: {left_foot_x} right_ankle_x: {right_ankle_x}")

    # --- 3스텝 (교차 안 함) 로직 ---
    lm3 = landmarks_dict[3]
    left_ankle_x_s3 = lm3[LEFT_ANKLE].x
    right_ankle_x_s3 = lm3[RIGHT_ANKLE].x

    if right_ankle_x_s3 > left_ankle_x_s3:
        feedback[3].append("[Foot] Good: Feet are uncrossed correctly for step 3.")
    else:
        feedback[3].append("[Foot] Advice: Feet should not be crossed in step 3.")

    return feedback

def visualize_torso_analysis(video_path: str, marked_steps: list, analysis_results: dict):
    """스텝 2, 3, 4, 5의 프레임 위에 몸통 벡터와 각도를 그려 시각화합니다."""
    annotated_images = []
    angles = analysis_results.get('angles', {})
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return []

    mp_pose = mp.solutions.pose
    LEFT_SHOULDER = mp_pose.PoseLandmark.LEFT_SHOULDER
    RIGHT_SHOULDER = mp_pose.PoseLandmark.RIGHT_SHOULDER
    LEFT_HIP = mp_pose.PoseLandmark.LEFT_HIP
    RIGHT_HIP = mp_pose.PoseLandmark.RIGHT_HIP

    # 랜드마크와 색상 정의 (BGR 순서)
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

        for landmark_enum, color in LANDMARK_COLORS.items():
            landmark = image_landmarks.landmark[landmark_enum.value]
            cx = int(landmark.x * w)
            cy = int(landmark.y * h)
            cv2.circle(frame, (cx, cy), 10, color, -1)

        shoulder_l = image_landmarks.landmark[LEFT_SHOULDER]
        shoulder_r = image_landmarks.landmark[RIGHT_SHOULDER]
        hip_l = image_landmarks.landmark[LEFT_HIP]
        hip_r = image_landmarks.landmark[RIGHT_HIP]
        shoulder_mid_px = (int((shoulder_l.x + shoulder_r.x) * w / 2), int((shoulder_l.y + shoulder_r.y) * h / 2))
        hip_mid_px = (int((hip_l.x + hip_r.x) * w / 2), int((hip_l.y + hip_r.y) * h / 2))

        cv2.line(frame, hip_mid_px, shoulder_mid_px, (0, 255, 255), 3)
        cv2.circle(frame, hip_mid_px, 7, (0, 0, 255), -1)
        cv2.circle(frame, shoulder_mid_px, 7, (255, 0, 0), -1)
        
        angle = angles.get(step_num)
        if angle is not None:
            cv2.putText(frame, f"Step {step_num} Torso Tilt: {angle:.2f} deg", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        annotated_images.append(frame)

    cap.release()
    return annotated_images
