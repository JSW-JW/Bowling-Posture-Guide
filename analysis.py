import cv2
import numpy as np
from mediapipe.framework.formats import landmark_pb2

# --- 랜드마크 인덱스 정의 ---
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24

def get_landmark_point(landmarks: landmark_pb2.NormalizedLandmarkList, index: int, is_world=False) -> np.ndarray:
    """랜드마크 리스트에서 특정 인덱스의 랜드마크 좌표를 numpy 배열로 반환합니다."""
    if is_world:
        return np.array([landmarks.landmark[index].x, landmarks.landmark[index].y, landmarks.landmark[index].z])
    else:
        return np.array([landmarks.landmark[index].x, landmarks.landmark[index].y])

def calculate_torso_tilt_from_image(landmarks) -> float:
    """
    상체의 좌우 기울기를 계산합니다. (이미지 좌표 기준)
    반환값은 수직축과의 각도(도)이며, 오른쪽으로 기울수록 양수 값이 커집니다.
    """
    shoulder_l = get_landmark_point(landmarks, LEFT_SHOULDER)
    shoulder_r = get_landmark_point(landmarks, RIGHT_SHOULDER)
    hip_l = get_landmark_point(landmarks, LEFT_HIP)
    hip_r = get_landmark_point(landmarks, RIGHT_HIP)

    shoulder_mid_pt = (shoulder_l + shoulder_r) / 2
    hip_mid_pt = (hip_l + hip_r) / 2
    
    torso_vector = shoulder_mid_pt - hip_mid_pt
    
    # 오른쪽으로 기울수록 양수가 되도록 수정 (arctan2(x, -y) -> arctan2(-x, -y))
    angle = np.degrees(np.arctan2(-torso_vector[0], -torso_vector[1]))
    return angle

def analyze_torso_angle(marked_steps: list) -> dict:
    """
    지정된 스텝 데이터에서 상체 각도를 분석하고,
    피드백과 계산된 각도 값을 딕셔너리로 반환합니다.
    """
    feedback = []
    angles = {}
    
    try:
        landmarks_step3 = next(item['image_landmarks'] for item in marked_steps if item["step"] == 3)
        landmarks_step4 = next(item['image_landmarks'] for item in marked_steps if item["step"] == 4)
        landmarks_step5 = next(item['image_landmarks'] for item in marked_steps if item["step"] == 5)
    except (StopIteration, KeyError):
        feedback.append("Error: Image landmarks for steps 3, 4, 5 must be available.")
        return {"feedback": feedback, "angles": angles}

    angle_step3 = calculate_torso_tilt_from_image(landmarks_step3)
    angle_step4 = calculate_torso_tilt_from_image(landmarks_step4)
    angle_step5 = calculate_torso_tilt_from_image(landmarks_step5)
    angles = {3: angle_step3, 4: angle_step4, 5: angle_step5}

    # 피드백 생성 로직 (오른쪽으로 기울수록 각도가 커지는 기준)
    if angle_step4 > angle_step3 + 5:  # 5도 이상 더 오른쪽으로 기울어졌을 때 '좋음'
        feedback.append("[Step 3->4] Good: Torso is properly tilted to the right.")
    else:
        feedback.append("[Step 3->4] Advice: Try to tilt your torso more to the right in step 4.")

    if abs(angle_step5 - angle_step4) < 5:
        feedback.append("[Step 4->5] Good: Torso angle is well-maintained until release.")
    else:
        feedback.append("[Step 4->5] Advice: Try to maintain your torso angle from step 4 through the release in step 5.")
        
    return {"feedback": feedback, "angles": angles}

def visualize_torso_analysis(video_path: str, marked_steps: list, analysis_results: dict):
    """
    스텝 3, 4, 5의 프레임 위에 몸통 벡터와 각도를 그려 시각화하고,
    분석용 이미지 리스트를 반환합니다.
    """
    annotated_images = []
    angles = analysis_results.get('angles', {})

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video for visualization.")
        return []

    for step_num in [3, 4, 5]:
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
            cv2.putText(frame, f"Step {step_num} Torso Tilt: {angle:.2f} deg", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        annotated_images.append(frame)

    cap.release()
    return annotated_images
