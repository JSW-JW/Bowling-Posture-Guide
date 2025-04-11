from app.pose_estimator import PoseEstimator
from app.step_segmenter import detect_step_frames
from app.analyzer import analyze_step_1
import cv2
import json

pose_estimator = PoseEstimator()
cap = cv2.VideoCapture("static/sample_video.mp4")

landmarks_per_frame = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    landmarks = pose_estimator.estimate(frame)
    landmarks_per_frame.append(landmarks)

step_frames = detect_step_frames(landmarks_per_frame)
step_1_result = analyze_step_1(landmarks_per_frame[step_frames[0]])

with open("output/analysis_report.json", "w") as f:
    json.dump({"step_1_weight_balance": step_1_result}, f, indent=2)
