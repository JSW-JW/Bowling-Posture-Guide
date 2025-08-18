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
# 손목 관련 랜드마크 (오른손잡이 기준)
RIGHT_WRIST = 16
RIGHT_ELBOW = 14
RIGHT_INDEX = 20
RIGHT_PINKY = 18
LEFT_WRIST = 15
LEFT_ELBOW = 13
LEFT_INDEX = 19
LEFT_PINKY = 17

class ArmTracker:
    """연속성 기반 팔 인식 보정 클래스"""
    
    def __init__(self):
        self.arm_pattern_history = []  # 각 스텝별 팔 위치 패턴 저장
        self.confidence_threshold = 0.6  # 60% 이상 일치하면 해당 패턴으로 결정
        
    def analyze_arm_patterns(self, marked_steps):
        """4스텝에서만 팔 위치 패턴을 분석하여 오인식 여부 판단"""
        step_4_data = None
        step_4_is_correct = True  # 기본값: 정상
        
        # 4스텝 데이터 찾기
        for step_data in marked_steps:
            if step_data['step'] == 4:
                step_4_data = step_data
                break
                
        if step_4_data is not None:
            landmarks = step_4_data['image_landmarks'].landmark
            
            # MediaPipe의 LEFT/RIGHT 랜드마크 Y좌표 비교
            mediapipe_left_wrist_y = landmarks[LEFT_WRIST].y
            mediapipe_right_wrist_y = landmarks[RIGHT_WRIST].y
            
            # 화면의 최상단 기준 y=0, 위치가 높을수록 y 값은 작아짐.
            step_4_is_correct = mediapipe_right_wrist_y < mediapipe_left_wrist_y
        
        # 4스텝 결과만 저장
        self.step_4_correction_needed = not step_4_is_correct
        
        return {
            'step_4_found': step_4_data is not None,
            'step_4_is_correct': step_4_is_correct,
            'correction_needed': not step_4_is_correct
        }
    
    def get_corrected_arm_landmarks(self, landmarks, step_num):
        """해당 스텝에서 보정된 팔 랜드마크 반환"""
        if step_num == 4:
            # 4스텝에서만 오인식 검사 및 보정 적용
            if hasattr(self, 'step_4_correction_needed') and self.step_4_correction_needed:
                # 4스텝에서 오인식 감지됨 - LEFT를 RIGHT로 사용
                return LEFT_WRIST, LEFT_INDEX, LEFT_PINKY
            else:
                # 4스텝에서 정상 - MediaPipe 결과 그대로 사용
                return RIGHT_WRIST, RIGHT_INDEX, RIGHT_PINKY
        else:
            # 1,2,3,5스텝은 항상 MediaPipe 결과 그대로 사용
            return RIGHT_WRIST, RIGHT_INDEX, RIGHT_PINKY

# 전역 ArmTracker 인스턴스
arm_tracker = ArmTracker()

def get_landmark_point(landmarks: landmark_pb2.NormalizedLandmarkList, index: int) -> np.ndarray:
    """랜드마크 리스트에서 특정 인덱스의 2D 랜드마크 좌표(x, y)를 numpy 배열로 반환합니다."""
    return np.array([landmarks.landmark[index].x, landmarks.landmark[index].y])

def calculate_torso_tilt_from_image(landmarks) -> float:
    """Calculate torso tilt angle. Positive values indicate right tilt."""
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
    """Analyze torso angles and return step-wise feedback and angle values."""
    feedback = {2: [], 3: [], 4: [], 5: []}
    angles = {}
    try:
        landmarks_step2 = next(item['image_landmarks'] for item in marked_steps if item["step"] == 2)
        landmarks_step3 = next(item['image_landmarks'] for item in marked_steps if item["step"] == 3)
        landmarks_step4 = next(item['image_landmarks'] for item in marked_steps if item["step"] == 4)
        landmarks_step5 = next(item['image_landmarks'] for item in marked_steps if item["step"] == 5)
    except (StopIteration, KeyError):
        return {"feedback": {2:["Torso analysis error"], 3:["Torso analysis error"], 4:["Torso analysis error"], 5:["Torso analysis error"]}, "angles": angles}

    angle_step2 = calculate_torso_tilt_from_image(landmarks_step2)
    angle_step3 = calculate_torso_tilt_from_image(landmarks_step3)
    angle_step4 = calculate_torso_tilt_from_image(landmarks_step4)
    angle_step5 = calculate_torso_tilt_from_image(landmarks_step5)
    angles = {2: angle_step2, 3: angle_step3, 4: angle_step4, 5: angle_step5}

    if angle_step3 > angle_step2:
        feedback[3].append("[Torso] Good: Proper tilt. Step3 torso angle > Step 2 torso angle")
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

# Additional landmark indices
LEFT_KNEE = 25
RIGHT_KNEE = 26

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
        
        # Future: implement proper crossover ratio calculation
        # crossover_ratio = (right_ankle_x - left_ankle_x) / hip_width if hip_width > 0 else 0
        
        # if min_crossover_threshold <= crossover_ratio <= max_crossover_threshold:
        if left_foot_x + 0.08 < right_ankle_x:
            feedback[step_num].append(f"[Foot] Advice: Crossover needed more.")
        else:
            feedback[step_num].append(f"[Foot] Good: Crossover is sufficient.")

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
    """Visualize torso analysis on video frames for steps 2-5."""
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

def detect_ball_position(landmarks, step_num=None):
    """
    공의 위치를 추정합니다. 손목, 검지, 새끼손가락 위치를 조합하여 
    손바닥 중앙 근처의 공 위치를 더 정확하게 추정합니다.
    
    Args:
        landmarks: MediaPipe 랜드마크
        step_num: 현재 스텝 번호 (팔 보정용)
        
    Returns:
        tuple: (x, y) 공의 추정 위치
    """
    try:
        # 보정된 팔 랜드마크 사용
        if step_num is not None:
            wrist_idx, index_idx, pinky_idx = arm_tracker.get_corrected_arm_landmarks(landmarks, step_num)
        else:
            wrist_idx, index_idx, pinky_idx = RIGHT_WRIST, RIGHT_INDEX, RIGHT_PINKY
            
        wrist = get_landmark_point(landmarks, wrist_idx)
        index = get_landmark_point(landmarks, index_idx)
        pinky = get_landmark_point(landmarks, pinky_idx)
        
        # 손바닥 중심점 계산: 검지와 새끼손가락의 중점
        finger_center = (index + pinky) / 2
        
        # 손목에서 손가락 중심으로의 벡터 계산
        wrist_to_finger_vector = finger_center - wrist
        wrist_to_finger_distance = np.linalg.norm(wrist_to_finger_vector)
        
        if wrist_to_finger_distance > 0:
            # 손목-손가락 벡터에 수직인 방향 벡터 계산 (2D에서)
            # (x, y) 벡터의 수직 벡터는 (-y, x) 또는 (y, -x)
            perpendicular_vector = np.array([-wrist_to_finger_vector[1], wrist_to_finger_vector[0]])
            perpendicular_distance = np.linalg.norm(perpendicular_vector)
            
            if perpendicular_distance > 0:
                # 수직 방향으로 정규화
                perpendicular_unit_vector = perpendicular_vector / perpendicular_distance
                
                # 손목과 손가락 중점에서 수직 방향으로 손목-손가락 거리의 일정 비율만큼 이동
                base_position = (wrist + finger_center) / 2
                offset_distance = wrist_to_finger_distance * 0.3  # 30% 거리만큼 수직 이동
                
                # 공은 손바닥에서 약간 떨어진 위치 (카메라 쪽 또는 반대쪽)
                ball_position = base_position + perpendicular_unit_vector * offset_distance
            else:
                ball_position = (wrist + finger_center) / 2
        else:
            # fallback: 손목과 finger_center 중점
            ball_position = (wrist + finger_center) / 2
        
        return ball_position[0], ball_position[1]
    except:
        # fallback: 손목 위치 사용
        try:
            wrist = get_landmark_point(landmarks, RIGHT_WRIST)
            return wrist[0], wrist[1]
        except:
            return None, None

def analyze_wrist_turnout_by_ball_position(landmarks, step_num=None) -> dict:
    """
    공의 x축을 기준으로 검지손가락 위치를 비교하여 턴아웃을 분석합니다.
    검지손가락이 공의 x축보다 왼쪽에 있으면 턴아웃으로 판단합니다.
    
    Returns:
        dict: 분석 결과와 피드백
    """
    try:
        ball_x, ball_y = detect_ball_position(landmarks, step_num)
        
        # 보정된 팔 랜드마크 사용
        if step_num is not None:
            wrist_idx, index_idx, pinky_idx = arm_tracker.get_corrected_arm_landmarks(landmarks, step_num)
        else:
            wrist_idx, index_idx, pinky_idx = RIGHT_WRIST, RIGHT_INDEX, RIGHT_PINKY
            
        index_finger = get_landmark_point(landmarks, index_idx)
        
        if ball_x is None or ball_y is None:
            return {"turnout_detected": False, "feedback": "Ball position detection failed"}
        
        index_x = index_finger[0]
        
        # 검지손가락이 공보다 왼쪽에 있으면 턴아웃
        turnout_detected = index_x < ball_x
        
        offset = abs(ball_x - index_x)
        
        if turnout_detected:
            if offset > 0.05:  # 5% 이상 차이
                severity = "Alert"
                feedback = f"[Wrist] Alert: Excessive turn-out detected (offset: {offset:.3f})"
            else:
                severity = "Caution" 
                feedback = f"[Wrist] Caution: Slight turn-out detected (offset: {offset:.3f})"
        else:
            severity = "Good"
            feedback = f"[Wrist] Good: Proper wrist alignment maintained"
            
        return {
            "turnout_detected": turnout_detected,
            "severity": severity,
            "feedback": feedback,
            "ball_x": ball_x,
            "index_x": index_x,
            "offset": offset
        }
        
    except Exception as e:
        return {"turnout_detected": False, "feedback": f"Wrist turnout analysis error: {e}"}

def analyze_wrist_turnout(marked_steps: list) -> dict:
    """
    4스텝에서만 손목 턴아웃을 분석합니다.
    공의 x 좌표를 기준으로 검지손가락 위치를 비교하여 턴아웃을 감지합니다.
    
    Args:
        marked_steps: 각 스텝의 랜드마크 데이터가 포함된 리스트
        
    Returns:
        dict: 4스텝 피드백과 분석 결과가 포함된 딕셔너리
    """
    feedback = {4: []}
    analysis_data = {}
    
    try:
        # 1. 팔 패턴 분석 수행 (4 스텝 대상)
        arm_analysis = arm_tracker.analyze_arm_patterns(marked_steps)
        
        print(f"[DEBUG] Arm Pattern Analysis:")
        print(f"  - Step 4 found: {arm_analysis['step_4_found']}")
        print(f"  - Step 4 correct: {arm_analysis['step_4_is_correct']}")
        print(f"  - Correction needed: {arm_analysis['correction_needed']}")
        
        # 2. 4스텝 랜드마크 추출
        step_4_data = next(item for item in marked_steps if item["step"] == 4)
        landmarks = step_4_data['image_landmarks']
        
        # 3. 턴아웃 분석 (보정된 팔 사용)
        result = analyze_wrist_turnout_by_ball_position(landmarks, step_num=4)
        
        feedback[4].append(result["feedback"])
        
        # 4. 추가 조언 제공
        if result["turnout_detected"] and result.get("offset", 0) > 0.05:
            feedback[4].append("[Wrist] Advice: Keep index finger aligned with or right of ball center")
            feedback[4].append("[Wrist] Advice: Focus on maintaining neutral wrist position during backswing")
        
        # 5. 팔 보정 정보 추가
        if arm_analysis['correction_needed']:
            feedback[4].append("[System] Note: Arm recognition corrected for step 4")
        
        analysis_data = result
        analysis_data['arm_analysis'] = arm_analysis
        
    except (StopIteration, KeyError):
        feedback[4].append("[Wrist] Error: Step 4 data not found for turnout analysis")
    except Exception as e:
        feedback[4].append(f"[Wrist] Error: {str(e)}")
    
    return {"feedback": feedback, "analysis_data": analysis_data}
