import mediapipe as mp
import cv2

mp_pose = mp.solutions.pose

class PoseEstimator:
    def __init__(self):
        self.pose = mp_pose.Pose()

    def estimate(self, frame):
        results = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return results.pose_landmarks
