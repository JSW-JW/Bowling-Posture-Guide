import numpy as np
from mediapipe.python.solutions.pose import PoseLandmark

def compute_velocity(z_sequence):
    return np.gradient(z_sequence)

def compute_acceleration(velocity_sequence):
    return np.gradient(velocity_sequence)

def detect_step_segments_boosted(
    landmarks_sequence,
    min_step_frames=5,
    z_thresh=0.02,
    accel_thresh=0.005
):
    foot_labels = [
        ("1스텝", PoseLandmark.LEFT_ANKLE),
        ("2스텝", PoseLandmark.RIGHT_ANKLE),
        ("3스텝", PoseLandmark.LEFT_ANKLE),
        ("4스텝", PoseLandmark.RIGHT_ANKLE),
        ("5스텝", PoseLandmark.LEFT_ANKLE),
    ]
    
    step_segments = []
    current_step = 0
    in_motion = False
    motion_start = 0
    num_frames = len(landmarks_sequence)
    
    if num_frames < 2:
        return step_segments

    while current_step < len(foot_labels):
        step_name, foot_index = foot_labels[current_step]
        
        z_seq = [frame[foot_index.value].z for frame in landmarks_sequence]
        velocity = compute_velocity(z_seq)
        acceleration = compute_acceleration(velocity)

        for i in range(1, num_frames):
            v = abs(velocity[i])
            a = abs(acceleration[i])
            
            if not in_motion:
                if v > z_thresh:
                    in_motion = True
                    motion_start = i
            else:
                if v < z_thresh and a < accel_thresh:
                    duration = i - motion_start
                    if duration >= min_step_frames:
                        step_segments.append((step_name, motion_start, i))
                        current_step += 1
                        in_motion = False
                        break
                elif i == num_frames - 1 and in_motion:
                    step_segments.append((step_name, motion_start, i))
                    current_step += 1
                    in_motion = False

        if current_step >= len(foot_labels):
            break

    return step_segments