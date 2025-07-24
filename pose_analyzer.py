import cv2
import mediapipe as mp
import time
from analysis import analyze_torso_angle # analysis.py에서 함수 임포트

# Mediapipe Pose 모델 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True,
                    model_complexity=2,
                    enable_segmentation=True,
                    min_detection_confidence=0.5)

def user_driven_step_segmentation(video_path):
    """
    사용자가 's' 키를 눌러 각 스텝의 프레임을 수동으로 지정하고,
    지정된 프레임의 자세 정보를 저장합니다.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return []

    step_data = []
    current_step = 1
    paused = False
    image_to_show = None

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
        cv2.putText(display_image, instruction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(display_image, f"Current Step: {current_step}/5", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
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
                results = pose.process(image_rgb)
                if results.pose_world_landmarks:
                    step_data.append({"step": current_step, "frame_number": frame_num, "landmarks": results.pose_world_landmarks})
                    print(f"Step {current_step} marked at frame {frame_num}.")
                    current_step += 1
                else:
                    print(f"Could not detect pose for Step {current_step} at frame {frame_num}. Please try another frame.")
            if current_step > 5:
                paused = True

    cap.release()
    cv2.destroyAllWindows()
    print("\n--- Step Segmentation Summary ---")
    for data in step_data:
        print(f"Step {data['step']}: Frame {data['frame_number']}")
    print("---------------------------------")
    return step_data

def replay_step_segments(video_path, steps_data, analysis_results):
    """
    사용자가 'n'키를 누를 때마다 다음 스텝 구간을 재생하며, 분석 데이터를 화면에 표시합니다.
    """
    if not steps_data:
        print("No steps were marked. Cannot replay.")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return

    angles = analysis_results.get('angles', {})
    
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
            
            last_frame_in_segment = frame.copy() # 마지막 프레임 저장
            
            cv2.putText(frame, seg['label'], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
            
            step_angle = angles.get(seg['step_num'])
            if step_angle is not None:
                angle_text = f"Torso Angle (Step {seg['step_num']}): {step_angle:.2f} deg"
                cv2.putText(frame, angle_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow('Segment Replay', frame)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                print("Replay interrupted by user.")
                return
        
        # 구간 재생 후 마지막 프레임에서 대기
        if last_frame_in_segment is not None:
            cv2.putText(last_frame_in_segment, "Press 'n' for next, 'q' to quit", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
            cv2.imshow('Segment Replay', last_frame_in_segment)
            while True:
                key = cv2.waitKey(0) & 0xFF
                if key == ord('n'):
                    break
                elif key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    print("Replay interrupted by user.")
                    return
    
    cap.release()
    cv2.destroyAllWindows()
    print("Finished replaying all segments.")


if __name__ == '__main__':
    video_file = 'static/videos/IMG_1564.MP4'
    
    if video_file == 'path/to/your/video.mp4':
        print("Please update the 'video_file' variable in the script with the path to your video.")
    else:
        marked_steps = user_driven_step_segmentation(video_file)
        
        if marked_steps and len(marked_steps) == 5:
            torso_analysis_result = analyze_torso_angle(marked_steps)
            
            print("\nReplaying marked segments with analysis data...")
            replay_step_segments(video_file, marked_steps, torso_analysis_result)
            
            print("\n--- Posture Analysis Feedback ---")
            for feedback in torso_analysis_result.get("feedback", []):
                print(f"- {feedback}")
            print("---------------------------------")

        elif marked_steps:
            print(f"\nOnly {len(marked_steps)} steps were marked. Please mark all 5 steps to proceed.")
