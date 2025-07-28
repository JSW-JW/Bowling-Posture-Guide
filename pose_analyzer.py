import cv2
import mediapipe as mp
import time
from analysis import analyze_torso_angle, analyze_foot_position_from_image, visualize_torso_analysis

# Mediapipe Pose 모델 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True,
                    model_complexity=2,
                    enable_segmentation=True,
                    min_detection_confidence=0.5)

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
    print("\n--- Step Segmentation Summary ---")
    for data in step_data:
        print(f"Step {data['step']}: Frame {data['frame_number']}")
    print("---------------------------------")
    return step_data

def replay_step_segments(video_path, steps_data, torso_analysis, foot_analysis):
    """
    사용자가 'n'키를 누를 때마다 다음 스텝 구간을 재생하며,
    분석 데이터와 피드백을 화면에 직접 표시합니다.
    """
    if not steps_data:
        print("No steps were marked. Cannot replay.")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return

    angles = torso_analysis.get('angles', {})
    torso_feedback = torso_analysis.get('feedback', {})
    
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
            
            last_frame_in_segment = frame.copy()
            
            # --- 현재 프레임에 정보 표시 ---
            y_offset = 30
            # 1. 구간 라벨
            cv2.putText(frame, seg['label'], (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
            y_offset += 40
            
            # 2. 각도 수치
            step_angle = angles.get(seg['step_num'])
            if step_angle is not None:
                angle_text = f"Torso Angle: {step_angle:.2f} deg"
                cv2.putText(frame, angle_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
                y_offset += 40

            # 3. 피드백 텍스트
            current_step_feedback = torso_feedback.get(seg['step_num'], []) + foot_analysis.get(seg['step_num'], [])
            for text in current_step_feedback:
                color = (0, 255, 0) if "Good" in text else (0, 0, 255)
                cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
                y_offset += 30

            cv2.imshow('Segment Replay', frame)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return
        
        if last_frame_in_segment is not None:
            # --- 마지막 프레임에 최종 정보 표시 ---
            y_offset = 30
            cv2.putText(last_frame_in_segment, seg['label'], (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
            y_offset += 40
            step_angle = angles.get(seg['step_num'])
            if step_angle is not None:
                angle_text = f"Torso Angle: {step_angle:.2f} deg"
                cv2.putText(last_frame_in_segment, angle_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
                y_offset += 40
            
            current_step_feedback = torso_feedback.get(seg['step_num'], []) + foot_analysis.get(seg['step_num'], [])
            for text in current_step_feedback:
                color = (0, 255, 0) if "Good" in text else (0, 0, 255)
                cv2.putText(last_frame_in_segment, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
                y_offset += 30

            cv2.imshow('Segment Replay', last_frame_in_segment)
            cv2.waitKey(2000)

            cv2.putText(last_frame_in_segment, "Press 'n' for next, 'q' to quit", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2, cv2.LINE_AA)
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
    video_file = 'static/videos/IMG_1564.MP4'
    
    if video_file == 'path/to/your/video.mp4':
        print("Please update the 'video_file' variable in the script with the path to your video.")
    else:
        marked_steps = user_driven_step_segmentation(video_file)
        
        if marked_steps and len(marked_steps) == 5:
            torso_analysis_result = analyze_torso_angle(marked_steps)
            foot_analysis_result = analyze_foot_position_from_image(marked_steps)
            
            print("\nReplaying marked segments with analysis data...")
            replay_step_segments(video_file, marked_steps, torso_analysis_result, foot_analysis_result)
            
            print("\n--- Final Posture Analysis Summary ---")
            all_feedback = {**torso_analysis_result.get("feedback", {}), **foot_analysis_result}
            for step in sorted(all_feedback.keys()):
                for feedback in all_feedback.get(step, []):
                    print(f"- Step {step}: {feedback}")
            print("------------------------------------")

            print("\nDisplaying analysis visualization...")
            annotated_images = visualize_torso_analysis(video_file, marked_steps, torso_analysis_result)
            for i, img in enumerate(annotated_images):
                cv2.imshow(f"Torso Analysis - Step {i+3}", img)
            
            print("Press any key to exit visualization.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        elif marked_steps:
            print(f"\nOnly {len(marked_steps)} steps were marked. Please mark all 5 steps to proceed.")