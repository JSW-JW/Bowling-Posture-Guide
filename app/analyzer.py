from mediapipe.python.solutions.pose import PoseLandmark

def analyze_step_1(frame_landmarks):
    right_foot_x = frame_landmarks.landmark[PoseLandmark.RIGHT_HEEL.value].x
    body_center_x = (
        frame_landmarks.landmark[PoseLandmark.LEFT_HIP.value].x +
        frame_landmarks.landmark[PoseLandmark.RIGHT_HIP.value].x
    ) / 2

    deviation = abs(right_foot_x - body_center_x)
    return deviation < 0.1
