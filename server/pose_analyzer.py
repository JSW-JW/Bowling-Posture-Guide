"""
Standalone Pose Analysis Tool

This module provides an interactive GUI tool for manual step segmentation 
and pose analysis of bowling videos. It's separate from the main FastAPI 
application and used for testing and development purposes.

Usage: Run this script directly to analyze bowling videos interactively.
"""

import cv2
import mediapipe as mp
import time
import numpy as np
from typing import Union, List, Dict, Any
from analysis import analyze_torso_angle, analyze_foot_crossover_by_x, visualize_torso_analysis

# Mediapipe Pose 모델 초기화
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
# 리플레이 시 동적 추적을 위해 static_image_mode=False 로 설정
pose_replay = mp_pose.Pose(static_image_mode=False, model_complexity=2, min_detection_confidence=0.5)

def user_driven_step_segmentation(video_path):
    """
    사용자가 's' 키를 눌러 각 스텝의 프레임을 수동으로 지정하고,
    월드 랜드마크와 이미지 랜드마크를 모두 저장합니다.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return []

    step_data = []
    current_step = 1
    paused = False
    image_to_show = None
    
    # 스텝 지정을 위한 별도의 Pose 인스턴스 (static_image_mode=True)
    pose_for_marking = mp_pose.Pose(static_image_mode=True, model_complexity=2)

    # 동적 폰트 크기를 위한 영상 높이 가져오기
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    font_scale = height / 1080.0  # 1080p를 기준으로 폰트 크기 조절
    line_thickness = max(1, int(font_scale * 2))

    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("Finished video or failed to read frame.")
                break
            image_to_show = frame.copy()

        instruction = f"Press 's' to mark Step {current_step}"
        if current_step > 5:
            instruction = "All steps marked. Press 'q' to finish."
        
        display_image = image_to_show.copy()

        cv2.putText(display_image, instruction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), line_thickness, cv2.LINE_AA)
        cv2.putText(display_image, f"Current Step: {current_step}/5", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), line_thickness, cv2.LINE_AA)
        cv2.imshow('Bowling Step Marking', display_image)

        key = cv2.waitKey(0 if paused else 30) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord('s'):
            if current_step <= 5:
                frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                image_rgb = cv2.cvtColor(image_to_show, cv2.COLOR_BGR2RGB)
                results = pose_for_marking.process(image_rgb)
                if results.pose_world_landmarks and results.pose_landmarks:
                    step_data.append({
                        "step": current_step,
                        "frame_number": frame_num,
                        "world_landmarks": results.pose_world_landmarks,
                        "image_landmarks": results.pose_landmarks
                    })
                    print(f"Step {current_step} marked at frame {frame_num}.")
                    current_step += 1
                else:
                    print(f"Could not detect pose for Step {current_step} at frame {frame_num}. Please try another frame.")
            if current_step > 5:
                paused = True

    cap.release()
    cv2.destroyAllWindows()
    pose_for_marking.close()
    print("\n--- Step Segmentation Summary ---")
    for data in step_data:
        print(f"Step {data['step']}: Frame {data['frame_number']}")
    print("---------------------------------")
    return step_data

# ==============================================================================
# ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
# NEW HELPER FUNCTION: 텍스트 자동 줄바꿈 함수 추가
# ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
# ==============================================================================
def draw_text_with_wrap(image, text, origin, font, font_scale, color, thickness, max_width):
    """
    OpenCV 이미지에 텍스트를 그리되, 지정된 최대 너비를 초과하면 자동으로 줄바꿈합니다.
    다음 텍스트를 그릴 시작 y좌표를 반환합니다.
    """
    x, y = origin
    words = text.split(' ')
    current_line = ''

    for word in words:
        # 현재 라인에 다음 단어를 추가했을 때의 너비를 계산
        test_line = f"{current_line} {word}".strip()
        (test_line_width, test_line_height), _ = cv2.getTextSize(test_line, font, font_scale, thickness)

        # 만약 너비가 최대치를 초과하면, 현재 라인을 그리고 새 라인을 시작
        if test_line_width > max_width and current_line:
            cv2.putText(image, current_line, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
            y += test_line_height + 5  # 다음 줄로 y좌표 이동 (5는 줄간격)
            current_line = word
        else:
            current_line = test_line

    # 루프가 끝난 후 마지막 라인 그리기
    if current_line:
        (_, last_line_height), _ = cv2.getTextSize(current_line, font, font_scale, thickness)
        cv2.putText(image, current_line, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
        y += last_line_height + 5
        
    return y

def replay_step_segments(video_path, steps_data, all_analysis):
    """
    사용자가 'n'키를 누를 때마다 다음 스텝 구간을 재생하며,
    랜드마크, 분석 데이터, 피드백, visibility를 화면에 직접 표시합니다.
    """
    if not steps_data:
        print("No steps were marked. Cannot replay.")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return
    
    # 추가된 부분: 랜드마크와 색상 정의 (BGR 순서)
    LANDMARK_COLORS = {
        mp_pose.PoseLandmark.LEFT_SHOULDER: (0, 0, 255),      # 빨강
        mp_pose.PoseLandmark.RIGHT_SHOULDER: (0, 165, 255),   # 주황
        mp_pose.PoseLandmark.LEFT_HIP: (0, 255, 255),         # 노랑
        mp_pose.PoseLandmark.RIGHT_HIP: (0, 255, 0),         # 초록
    }    

    # 동적 폰트 크기 및 화면 너비 가져오기
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    font_scale = height / 1080.0
    line_thickness = max(1, int(font_scale * 2))

    angles = all_analysis.get('torso', {}).get('angles', {})
    
    segments = []
    start_frame = 0
    for i, step_info in enumerate(steps_data):
        end_frame = step_info['frame_number']
        label = f"Replaying: Start to Step {i+1}" if i == 0 else f"Replaying: Step {i} to Step {i+1}"
        segments.append({"start": start_frame, "end": end_frame, "label": label, "step_num": i + 1})
        start_frame = end_frame

    for seg in segments:
        last_frame_in_segment = None
        cap.set(cv2.CAP_PROP_POS_FRAMES, seg['start'])
        while cap.get(cv2.CAP_PROP_POS_FRAMES) < seg['end']:
            ret, frame = cap.read()
            if not ret:
                break
            
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose_replay.process(image_rgb)
            annotated_image = frame.copy()
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(245,117,66), thickness=line_thickness, circle_radius=line_thickness),
                                          mp_drawing.DrawingSpec(color=(245,66,230), thickness=line_thickness, circle_radius=line_thickness))
            
                landmarks = results.pose_landmarks.landmark
                for landmark_enum, color in LANDMARK_COLORS.items():
                    # 랜드마크의 정규화된 좌표를 실제 픽셀 좌표로 변환
                    cx = int(landmarks[landmark_enum.value].x * width)
                    cy = int(landmarks[landmark_enum.value].y * height)
                    # OpenCV를 사용하여 원(점) 그리기
                    cv2.circle(annotated_image, (cx, cy), max(5, line_thickness * 5), color, -1)

            y_offset = int(30 * font_scale)
            cv2.putText(annotated_image, seg['label'], (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 0), line_thickness, cv2.LINE_AA)
            y_offset += int(40 * font_scale)
            
            step_angle = angles.get(seg['step_num'])
            if step_angle is not None:
                angle_text = f"Torso Angle: {step_angle:.2f} deg"
                cv2.putText(annotated_image, angle_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), line_thickness, cv2.LINE_AA)
                y_offset += int(40 * font_scale)

            if results.pose_landmarks:
                left_ankle_vis = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].visibility
                right_ankle_vis = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].visibility
                vis_text = f"L-Ankle Vis: {left_ankle_vis:.2f} | R-Ankle Vis: {right_ankle_vis:.2f}"
                cv2.putText(annotated_image, vis_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, (255, 255, 255), line_thickness, cv2.LINE_AA)
                y_offset += int(30 * font_scale)

            current_step_feedback = all_analysis.get('torso',{}).get('feedback',{}).get(seg['step_num'], []) + \
                                    all_analysis.get('foot',{}).get(seg['step_num'], [])
            
            # ==============================================================================
            # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
            # MODIFICATION: 기존 putText를 새로운 줄바꿈 함수로 대체
            # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
            # ==============================================================================
            for text in current_step_feedback:
                color = (0, 255, 0) if "Good" in text else (0, 0, 255)
                y_offset = draw_text_with_wrap(
                    image=annotated_image,
                    text=text,
                    origin=(10, y_offset),
                    font=cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale=font_scale * 0.8,
                    color=color,
                    thickness=line_thickness,
                    max_width=width - 20  # 화면 좌우 10px씩 여백을 줌
                )
            
            last_frame_in_segment = annotated_image.copy()
            cv2.imshow('Segment Replay', annotated_image)

            if cv2.waitKey(30) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return
        
        if last_frame_in_segment is not None:
            cv2.imshow('Segment Replay', last_frame_in_segment)
            cv2.waitKey(2000)
            
            final_y_pos = last_frame_in_segment.shape[0] - int(30 * font_scale)
            cv2.putText(last_frame_in_segment, "Press 'n' for next, 'q' to quit", (10, final_y_pos), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 255), line_thickness, cv2.LINE_AA)
            cv2.imshow('Segment Replay', last_frame_in_segment)
            
            while True:
                key = cv2.waitKey(0) & 0xFF
                if key == ord('n'): break
                elif key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return
    
    cap.release()
    cv2.destroyAllWindows()
    print("Finished replaying all segments.")


if __name__ == '__main__':
    video_file = 'static/videos/신승현프로_2.mp4'
    
    if video_file == 'path/to/your/video.mp4':
        print("Please update the 'video_file' variable in the script with the path to your video.")
    else:
        marked_steps = user_driven_step_segmentation(video_file)
        
        if marked_steps and len(marked_steps) == 5:
            all_analysis_results = {
                "torso": analyze_torso_angle(marked_steps),
                "foot": analyze_foot_crossover_by_x(marked_steps),
            }
            
            print("\nReplaying marked segments with analysis data...")
            replay_step_segments(video_file, marked_steps, all_analysis_results)
            
            print("\n--- Final Posture Analysis Summary ---")
            torso_feedback = all_analysis_results['torso'].get('feedback', {})
            foot_feedback = all_analysis_results['foot']
            
            all_feedback_merged = {**torso_feedback, **foot_feedback}
            for step in sorted(all_feedback_merged.keys()):
                feedback_lists = [
                    torso_feedback.get(step, []),
                    foot_feedback.get(step, []),
                ]
                for feedback_list in feedback_lists:
                    for feedback in feedback_list:
                        print(f"- Step {step}: {feedback}")
            print("------------------------------------")

            print("\nDisplaying analysis visualization...")
            annotated_images = visualize_torso_analysis(video_file, marked_steps, all_analysis_results['torso'])
            for i, img in enumerate(annotated_images):
                cv2.imshow(f"Torso Analysis - Step {i+2}", img)
            
            print("Press any key to exit visualization.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        elif marked_steps:
            print(f"\nOnly {len(marked_steps)} steps were marked. Please mark all 5 steps to proceed.")
    
    pose_replay.close()