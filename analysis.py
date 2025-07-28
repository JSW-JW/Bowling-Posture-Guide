import cv2
import numpy as np
from mediapipe.framework.formats import landmark_pb2

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
    angle = np.degrees(np.arctan2(-torso_vector[0], -torso_vector[1]))
    return angle

def analyze_torso_angle(marked_steps: list) -> dict:
    """상체 각도를 분석하고 스텝별 피드백과 각도 값을 반환합니다."""
    feedback = {3: [], 4: [], 5: []}
    angles = {}
    try:
        landmarks_step3 = next(item['image_landmarks'] for item in marked_steps if item["step"] == 3)
        landmarks_step4 = next(item['image_landmarks'] for item in marked_steps if item["step"] == 4)
        landmarks_step5 = next(item['image_landmarks'] for item in marked_steps if item["step"] == 5)
    except (StopIteration, KeyError):
        return {"feedback": {3:["Torso analysis error"], 4:["Torso analysis error"], 5:["Torso analysis error"]}, "angles": angles}

    angle_step3 = calculate_torso_tilt_from_image(landmarks_step3)
    angle_step4 = calculate_torso_tilt_from_image(landmarks_step4)
    angle_step5 = calculate_torso_tilt_from_image(landmarks_step5)
    angles = {3: angle_step3, 4: angle_step4, 5: angle_step5}

    if angle_step4 > angle_step3 + 5:
        feedback[4].append("[Torso] Good: Proper tilt.")
    else:
        feedback[4].append("[Torso] Advice: Tilt more.")
    if abs(angle_step5 - angle_step4) < 5:
        feedback[5].append("[Torso] Good: Angle maintained.")
    else:
        feedback[5].append("[Torso] Advice: Maintain angle.")
    return {"feedback": feedback, "angles": angles}

def analyze_foot_position_from_image(marked_steps: list) -> dict:
    """
    발의 교차 여부를 x축 좌표로 분석하고 스텝별 피드백을 반환합니다.
    (오른손잡이 기준)
    """
    feedback = {2: [], 3: [], 4: []}
    try:
        landmarks = {s['step']: s['image_landmarks'] for s in marked_steps}
    except (StopIteration, KeyError):
        return {s: ["Foot analysis error"] for s in feedback.keys()}

    # --- 2 스텝: 정교한 교차 ---
    left_foot_index_x = get_landmark_point(landmarks[2], LEFT_FOOT_INDEX)[0]
    right_foot_index_x = get_landmark_point(landmarks[2], RIGHT_FOOT_INDEX)[0]
    left_ankle_x = get_landmark_point(landmarks[2], LEFT_ANKLE)[0]
    
    if right_foot_index_x < left_foot_index_x and right_foot_index_x > left_ankle_x:
         feedback[2].append("[Foot] Good: Crossover is ideal.")
    else:
         feedback[2].append("[Foot] Advice: Adjust crossover.")

    # --- 3 스텝: 교차 안 함 ---
    foot_x_step3_left = get_landmark_point(landmarks[3], LEFT_ANKLE)[0]
    foot_x_step3_right = get_landmark_point(landmarks[3], RIGHT_ANKLE)[0]
    if foot_x_step3_right > foot_x_step3_left:
        feedback[3].append("[Foot] Good: Feet uncrossed.")
    else:
        feedback[3].append("[Foot] Advice: Uncross feet.")

    # --- 4 스텝: 교차 함 ---
    foot_x_step4_left = get_landmark_point(landmarks[4], LEFT_ANKLE)[0]
    foot_x_step4_right = get_landmark_point(landmarks[4], RIGHT_ANKLE)[0]
    if foot_x_step4_right < foot_x_step4_left:
        feedback[4].append("[Foot] Good: Crossover correct.")
    else:
        feedback[4].append("[Foot] Advice: Crossover needed.")

    return feedback

def visualize_torso_analysis(video_path: str, marked_steps: list, analysis_results: dict):
    """스텝 3, 4, 5의 프레임 위에 몸통 벡터와 각도를 그려 시각화합니다."""
    annotated_images = []
    angles = analysis_results.get('angles', {})
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return []

    for step_num in [3, 4, 5]:
        try:
            step_info = next(item for item in marked_steps if item["step"] == step_num)
            frame_num = step_info['frame_number']
            image_landmarks = step_info['image_landmarks']
        except (StopIteration, KeyError): continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)
        ret, frame = cap.read()
        if not ret: continue
        
        h, w, _ = frame.shape
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
