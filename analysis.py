import numpy as np
from mediapipe.framework.formats import landmark_pb2

def get_landmark_point(landmarks: landmark_pb2.NormalizedLandmarkList, index: int) -> np.ndarray:
    """랜드마크 리스트에서 특정 인덱스의 랜드마크 좌표를 numpy 배열로 반환합니다."""
    return np.array([landmarks.landmark[index].x, landmarks.landmark[index].y, landmarks.landmark[index].z])

def calculate_torso_tilt(landmarks) -> float:
    """
    상체의 좌우 기울기를 계산합니다.
    반환값은 수직축과의 각도(도)이며, 오른쪽으로 기울수록 음수 값이 커집니다.
    """
    # Mediapipe Pose 랜드마크 인덱스
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_HIP = 23
    RIGHT_HIP = 24

    # 어깨와 엉덩이의 중간점 계산
    shoulder_mid_pt = (get_landmark_point(landmarks, LEFT_SHOULDER) + get_landmark_point(landmarks, RIGHT_SHOULDER)) / 2
    hip_mid_pt = (get_landmark_point(landmarks, LEFT_HIP) + get_landmark_point(landmarks, RIGHT_HIP)) / 2

    # 상체 벡터 계산 (엉덩이 -> 어깨)
    torso_vector = shoulder_mid_pt - hip_mid_pt

    # XY 평면(카메라 평면)에서의 각도 계산
    # atan2(x, y)를 사용하여 수직축(Y축)과의 각도를 구합니다.
    # Mediapipe의 Y축은 아래를 향하므로, -y를 사용하여 위쪽을 기준으로 각도를 계산합니다.
    # x가 양수(왼쪽 기울기)일 때 양의 각도, x가 음수(오른쪽 기울기)일 때 음의 각도가 나옵니다.
    angle = np.degrees(np.arctan2(torso_vector[0], -torso_vector[1]))
    return angle

def analyze_torso_angle(marked_steps: list) -> dict:
    """
    지정된 스텝 데이터에서 상체 각도를 분석하고,
    피드백과 계산된 각도 값을 딕셔너리로 반환합니다.
    """
    feedback = []
    angles = {}
    
    try:
        landmarks_step3 = next(item for item in marked_steps if item["step"] == 3)['landmarks']
        landmarks_step4 = next(item for item in marked_steps if item["step"] == 4)['landmarks']
        landmarks_step5 = next(item for item in marked_steps if item["step"] == 5)['landmarks']
    except StopIteration:
        feedback.append("Error: Steps 3, 4, and 5 must all be marked for torso analysis.")
        return {"feedback": feedback, "angles": angles}

    # 각 스텝의 상체 각도(몸통 기울기) 계산 및 저장
    angle_step3 = calculate_torso_tilt(landmarks_step3)
    angle_step4 = calculate_torso_tilt(landmarks_step4)
    angle_step5 = calculate_torso_tilt(landmarks_step5)
    angles = {3: angle_step3, 4: angle_step4, 5: angle_step5}

    # 피드백 생성 로직 (오른손잡이 기준)
    # 오른쪽 기울기는 음수. 더 기울이면 더 작은(음수) 값이 됨.
    if angle_step4 < angle_step3 - 5:  # 5도 이상 더 오른쪽으로 기울어졌을 때 '좋음'으로 판단
        feedback.append("[Step 3->4] Good: Torso is properly tilted to the right.")
    else:
        feedback.append("[Step 3->4] Advice: Try to tilt your torso more to the right in step 4.")

    # 4->5 스텝: 4스텝의 각도를 릴리즈까지 유지해야 함 (각도 차이가 5도 미만)
    if abs(angle_step5 - angle_step4) < 5:
        feedback.append("[Step 4->5] Good: Torso angle is well-maintained until release.")
    else:
        feedback.append("[Step 4->5] Advice: Try to maintain your torso angle from step 4 through the release in step 5.")
        
    return {"feedback": feedback, "angles": angles}
