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
from analysis import analyze_torso_angle, analyze_foot_crossover_by_x, visualize_torso_analysis, analyze_wrist_turnout, analyze_wrist_turnout_by_ball_position, detect_ball_position, arm_tracker

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

# ==============================================================================
# 디버그 모드 전용 함수들 (콜백으로 사용)
# ==============================================================================

def debug_ui_renderer(annotated_image, debug_info):
    """디버그 UI 렌더링 전용 함수"""
    width, height = debug_info['width'], debug_info['height']
    font_scale = debug_info['font_scale']
    line_thickness = debug_info['line_thickness']
    
    # 프레임 정보 표시
    absolute_frame = debug_info['absolute_frame']
    total_frames = debug_info['total_frames']
    fps = debug_info['fps']
    seg_label = debug_info['seg_label']
    current_segment_idx = debug_info['current_segment_idx']
    segment_count = debug_info['segment_count']
    current_frame_in_segment = debug_info['current_frame_in_segment']
    segment_length = debug_info['segment_length']
    
    y_offset = int(30 * font_scale)
    debug_info_text = f"{seg_label} | Frame: {absolute_frame}/{total_frames-1} ({absolute_frame/fps:.2f}s)"
    cv2.putText(annotated_image, debug_info_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, (255, 255, 0), line_thickness, cv2.LINE_AA)
    y_offset += int(35 * font_scale)
    
    segment_info = f"Segment {current_segment_idx + 1}/{segment_count} | Frame {current_frame_in_segment + 1}/{segment_length}"
    cv2.putText(annotated_image, segment_info, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.6, (200, 200, 200), line_thickness, cv2.LINE_AA)
    
    # 하단 조작 안내
    controls_y = height - 60
    cv2.putText(annotated_image, "←/→(A/D):1frame ↑/↓(W/S):10frames SPACE:play/pause N:next Q:quit", 
               (10, controls_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.5, (0, 255, 255), line_thickness, cv2.LINE_AA)
    
    # 자동재생 상태 표시
    auto_play = debug_info.get('auto_play', False)
    play_status = "AUTO-PLAY" if auto_play else "PAUSED"
    status_color = (0, 255, 0) if auto_play else (0, 0, 255)
    cv2.putText(annotated_image, f"[{play_status}]", (width - 120, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.6, status_color, line_thickness, cv2.LINE_AA)
    
    return annotated_image

def debug_landmark_renderer(annotated_image, results, debug_info):
    """디버그용 상세 랜드마크 정보 표시"""
    if not results.pose_landmarks:
        return annotated_image
        
    landmarks = results.pose_landmarks.landmark
    font_scale = debug_info['font_scale']
    line_thickness = debug_info['line_thickness']
    
    # 손목/검지 좌표 상세 표시
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    left_index = landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value]
    right_index = landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value]
    
    # 볼 위치 계산
    ball_x, ball_y = detect_ball_position(results.pose_landmarks, step_num=debug_info.get('step_num', 4))
    
    coord_texts = [
        f"L_Wrist: ({left_wrist.x:.3f}, {left_wrist.y:.3f}) vis:{left_wrist.visibility:.2f}",
        f"R_Wrist: ({right_wrist.x:.3f}, {right_wrist.y:.3f}) vis:{right_wrist.visibility:.2f}",
        f"L_Index: ({left_index.x:.3f}, {left_index.y:.3f}) vis:{left_index.visibility:.2f}",
        f"R_Index: ({right_index.x:.3f}, {right_index.y:.3f}) vis:{right_index.visibility:.2f}",
        f"Ball: ({ball_x:.3f}, {ball_y:.3f})" if ball_x is not None else "Ball: (None, None)"
    ]
    
    y_offset = debug_info.get('y_offset', int(120 * font_scale))
    for coord_text in coord_texts:
        cv2.putText(annotated_image, coord_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.5, (255, 255, 255), line_thickness, cv2.LINE_AA)
        y_offset += int(25 * font_scale)
    
    return annotated_image

def debug_input_handler(key, current_frame_in_segment, segment_length, auto_play):
    """디버그 입력 처리 전용 함수"""
    action = None
    new_frame = current_frame_in_segment
    new_auto_play = auto_play
    
    if key == ord('q') or key == 27:  # 'q' or ESC
        action = 'quit'
    elif key == 83 or key == ord('d'):  # Right arrow or 'd' - 1프레임 앞으로
        new_frame = min(current_frame_in_segment + 1, segment_length - 1)
        action = 'continue'
    elif key == 81 or key == ord('a'):  # Left arrow or 'a' - 1프레임 뒤로
        new_frame = max(current_frame_in_segment - 1, 0)
        action = 'continue'
    elif key == 82 or key == ord('w'):  # Up arrow or 'w' - 10프레임 앞으로
        new_frame = min(current_frame_in_segment + 10, segment_length - 1)
        action = 'continue'
    elif key == 84 or key == ord('s'):  # Down arrow or 's' - 10프레임 뒤로
        new_frame = max(current_frame_in_segment - 10, 0)
        action = 'continue'
    elif key == ord('n'):  # 'n' - 다음 세그먼트로
        action = 'next_segment'
    elif key == ord(' '):  # SPACE - 자동 재생 토글
        new_auto_play = not auto_play
        if new_auto_play:
            print("Auto-play enabled")
        else:
            print("Auto-play disabled")
        action = 'continue'
    
    return {
        'action': action,
        'frame': new_frame,
        'auto_play': new_auto_play
    }

def replay_step_segments(video_path, steps_data, all_analysis, 
                        ui_renderer=None, landmark_renderer=None, input_handler=None):
    """
    사용자가 'n'키를 누를 때마다 다음 스텝 구간을 재생하며,
    랜드마크, 분석 데이터, 피드백, visibility를 화면에 직접 표시합니다.
    
    Args:
        ui_renderer: 디버그 UI 렌더링 콜백 함수 (선택적)
        landmark_renderer: 상세 랜드마크 정보 표시 콜백 함수 (선택적)  
        input_handler: 입력 처리 콜백 함수 (선택적)
    """
    if not steps_data:
        print("No steps were marked. Cannot replay.")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return
    
    # 기본 설정
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    font_scale = height / 1080.0
    line_thickness = max(1, int(font_scale * 2))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # 랜드마크 색상 정의
    LANDMARK_COLORS = {
        mp_pose.PoseLandmark.LEFT_SHOULDER: (0, 0, 255),      # 빨강
        mp_pose.PoseLandmark.RIGHT_SHOULDER: (0, 165, 255),   # 주황
        mp_pose.PoseLandmark.LEFT_HIP: (0, 255, 255),         # 노랑
        mp_pose.PoseLandmark.RIGHT_HIP: (0, 255, 0),         # 초록
    }
    
    angles = all_analysis.get('torso', {}).get('angles', {})
    
    # 세그먼트 생성
    segments = []
    start_frame = 0
    for i, step_info in enumerate(steps_data):
        end_frame = step_info['frame_number']
        label = f"Replaying: Start to Step {i+1}" if i == 0 else f"Replaying: Step {i} to Step {i+1}"
        segments.append({"start": start_frame, "end": end_frame, "label": label, "step_num": i + 1})
        start_frame = end_frame
    
    # 콜백 함수가 있을 때는 디버그 모드로 처리
    is_debug_mode = input_handler is not None
    current_frame_in_segment = 0
    auto_play = not is_debug_mode
    
    for seg_idx, seg in enumerate(segments):
        segment_start = seg['start']
        segment_end = seg['end']
        segment_length = segment_end - segment_start
        
        if is_debug_mode:
            current_frame_in_segment = 0
        
        while True:
            # 프레임 위치 설정
            if is_debug_mode:
                absolute_frame = segment_start + current_frame_in_segment
                if absolute_frame >= segment_end:
                    break
                cap.set(cv2.CAP_PROP_POS_FRAMES, absolute_frame)
            else:
                if cap.get(cv2.CAP_PROP_POS_FRAMES) >= segment_end:
                    break
            
            ret, frame = cap.read()
            if not ret:
                break
            
            # MediaPipe 분석
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose_replay.process(image_rgb)
            annotated_image = frame.copy()
            
            # 기본 랜드마크 및 연결선 그리기
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=line_thickness, circle_radius=line_thickness),
                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=line_thickness, circle_radius=line_thickness)
                )
                
                landmarks = results.pose_landmarks.landmark
                
                # 기본 랜드마크 표시 (어깨, 엉덩이)
                for landmark_enum, color in LANDMARK_COLORS.items():
                    cx = int(landmarks[landmark_enum.value].x * width)
                    cy = int(landmarks[landmark_enum.value].y * height)
                    cv2.circle(annotated_image, (cx, cy), max(4, line_thickness * 3), color, -1)
                
                # 3, 4스텝에서 손목/검지 표시
                if seg['step_num'] in [3, 4]:
                    _draw_wrist_landmarks(annotated_image, landmarks, width, height, line_thickness, seg['step_num'])
            
            # UI 렌더링
            y_offset = _render_basic_ui(annotated_image, seg, angles, font_scale, line_thickness)
            
            # 콜백 함수 호출
            if ui_renderer and is_debug_mode:
                debug_info = _create_debug_info(seg_idx, seg, segments, current_frame_in_segment, 
                                              segment_length, absolute_frame if is_debug_mode else None,
                                              total_frames, fps, width, height, font_scale, line_thickness, auto_play)
                annotated_image = ui_renderer(annotated_image, debug_info)
            
            if landmark_renderer and is_debug_mode:
                debug_info = {'font_scale': font_scale, 'line_thickness': line_thickness, 
                            'step_num': seg['step_num'], 'y_offset': y_offset}
                annotated_image = landmark_renderer(annotated_image, results, debug_info)
            
            # 피드백 표시
            _render_feedback(annotated_image, all_analysis, seg['step_num'], y_offset, font_scale, line_thickness, width)
            
            cv2.imshow('Segment Replay', annotated_image)
            
            # 입력 처리
            if input_handler and is_debug_mode:
                key = cv2.waitKey(0) & 0xFF
                result = input_handler(key, current_frame_in_segment, segment_length, auto_play)
                
                if result['action'] == 'quit':
                    cap.release()
                    cv2.destroyAllWindows()
                    return
                elif result['action'] == 'next_segment':
                    break
                elif result['action'] == 'continue':
                    current_frame_in_segment = result['frame']
                    auto_play = result['auto_play']
                    if auto_play:
                        current_frame_in_segment = min(current_frame_in_segment + 1, segment_length - 1)
                        if current_frame_in_segment >= segment_length - 1:
                            break
            else:
                # 일반 모드 입력 처리
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return
        
        # 일반 모드에서는 세그먼트 완료 후 대기
        if not is_debug_mode:
            _show_segment_complete_screen(cap, segments, seg_idx, font_scale, line_thickness)
    
    cap.release()
    cv2.destroyAllWindows()
    print("Finished replaying all segments.")

# ==============================================================================
# 헬퍼 함수들
# ==============================================================================

def _draw_wrist_landmarks(annotated_image, landmarks, width, height, line_thickness, step_num):
    """손목과 검지 랜드마크 그리기"""
    # MediaPipe 호환 객체 생성
    class MockLandmarkList:
        def __init__(self, landmarks):
            self.landmark = landmarks
    
    class MockPoseResults:
        def __init__(self, landmarks):
            self.pose_landmarks = MockLandmarkList(landmarks)
    
    mock_results = MockPoseResults(landmarks)
    
    # 턴아웃 분석 수행 (보정된 팔 사용)
    wrist_analysis = analyze_wrist_turnout_by_ball_position(mock_results.pose_landmarks, step_num=step_num)
    
    # 보정된 결과에 따라 올바른 위치에 랜드마크 표시
    corrected_landmarks = arm_tracker.get_corrected_arm_landmarks(mock_results.pose_landmarks, step_num)
    correction_applied = corrected_landmarks[0] == 15  # LEFT_WRIST = 15 (보정 적용됨)
    
    if correction_applied:
        # MediaPipe가 오인식한 경우: Y좌표 기준으로 실제 위치 파악
        left_wrist_y = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y
        right_wrist_y = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y
        
        if right_wrist_y < left_wrist_y:
            # MediaPipe RIGHT가 실제 위쪽(실제 오른팔)
            _draw_corrected_landmarks(annotated_image, landmarks, width, height, line_thickness, 
                                    mp_pose.PoseLandmark.RIGHT_WRIST, mp_pose.PoseLandmark.RIGHT_INDEX, "R", (0, 165, 255))
            _draw_corrected_landmarks(annotated_image, landmarks, width, height, line_thickness,
                                    mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.LEFT_INDEX, "L", (255, 0, 0))
        else:
            # MediaPipe LEFT가 실제 위쪽(실제 오른팔)  
            _draw_corrected_landmarks(annotated_image, landmarks, width, height, line_thickness,
                                    mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.LEFT_INDEX, "R", (0, 165, 255))
            _draw_corrected_landmarks(annotated_image, landmarks, width, height, line_thickness,
                                    mp_pose.PoseLandmark.RIGHT_WRIST, mp_pose.PoseLandmark.RIGHT_INDEX, "L", (255, 0, 0))
    else:
        # MediaPipe가 올바르게 인식한 경우
        _draw_corrected_landmarks(annotated_image, landmarks, width, height, line_thickness,
                                mp_pose.PoseLandmark.RIGHT_WRIST, mp_pose.PoseLandmark.RIGHT_INDEX, "R", (0, 165, 255))
        _draw_corrected_landmarks(annotated_image, landmarks, width, height, line_thickness,
                                mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.LEFT_INDEX, "L", (255, 0, 0))
    
    # 볼 위치 표시 - 녹색
    ball_x, ball_y = detect_ball_position(mock_results.pose_landmarks, step_num=step_num)
    if ball_x is not None and ball_y is not None:
        ball_cx = int(ball_x * width)
        ball_cy = int(ball_y * height)
        cv2.circle(annotated_image, (ball_cx, ball_cy), max(4, line_thickness * 3), (0, 255, 0), -1)

def _draw_corrected_landmarks(annotated_image, landmarks, width, height, line_thickness, wrist_landmark, index_landmark, label, color):
    """보정된 랜드마크 그리기"""
    font_scale = height / 1080.0
    
    # 손목
    wrist = landmarks[wrist_landmark.value]
    wrist_cx = int(wrist.x * width)
    wrist_cy = int(wrist.y * height)
    cv2.circle(annotated_image, (wrist_cx, wrist_cy), max(4, line_thickness * 3), color, -1)
    cv2.putText(annotated_image, f"{label}-Wrist", (wrist_cx + 15, wrist_cy), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.7, color, line_thickness, cv2.LINE_AA)
    
    # 검지
    index = landmarks[index_landmark.value] 
    index_cx = int(index.x * width)
    index_cy = int(index.y * height)
    cv2.circle(annotated_image, (index_cx, index_cy), max(4, line_thickness * 3), color, -1)
    cv2.putText(annotated_image, f"{label}-Index", (index_cx + 15, index_cy), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.7, color, line_thickness, cv2.LINE_AA)

def _render_basic_ui(annotated_image, seg, angles, font_scale, line_thickness):
    """기본 UI 렌더링"""
    y_offset = int(30 * font_scale)
    cv2.putText(annotated_image, seg['label'], (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 0), line_thickness, cv2.LINE_AA)
    y_offset += int(40 * font_scale)
    
    step_angle = angles.get(seg['step_num'])
    if step_angle is not None:
        angle_text = f"Torso Angle: {step_angle:.2f} deg"
        cv2.putText(annotated_image, angle_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), line_thickness, cv2.LINE_AA)
        y_offset += int(40 * font_scale)
    
    return y_offset

def _create_debug_info(seg_idx, seg, segments, current_frame_in_segment, segment_length, absolute_frame, total_frames, fps, width, height, font_scale, line_thickness, auto_play):
    """디버그 정보 딕셔너리 생성"""
    return {
        'width': width, 'height': height, 'font_scale': font_scale, 'line_thickness': line_thickness,
        'absolute_frame': absolute_frame, 'total_frames': total_frames, 'fps': fps,
        'seg_label': seg['label'], 'current_segment_idx': seg_idx, 'segment_count': len(segments),
        'current_frame_in_segment': current_frame_in_segment, 'segment_length': segment_length,
        'auto_play': auto_play, 'step_num': seg['step_num']
    }

def _render_feedback(annotated_image, all_analysis, step_num, y_offset, font_scale, line_thickness, width):
    """피드백 텍스트 렌더링"""
    current_step_feedback = all_analysis.get('torso',{}).get('feedback',{}).get(step_num, []) + \
                            all_analysis.get('foot',{}).get(step_num, [])
    
    # 3스텝과 4스텝에서 손목 피드백 추가
    if step_num in [3, 4]:
        current_step_feedback.extend(all_analysis.get('wrist',{}).get('feedback',{}).get(step_num, []))
    
    for text in current_step_feedback:
        color = (0, 255, 0) if "Good" in text else (0, 0, 255)
        y_offset = draw_text_with_wrap(
            image=annotated_image, text=text, origin=(10, y_offset),
            font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=font_scale * 0.8, color=color,
            thickness=line_thickness, max_width=width - 20
        )

def _show_segment_complete_screen(cap, segments, seg_idx, font_scale, line_thickness):
    """세그먼트 완료 화면 표시 (일반 모드)"""
    # 마지막 프레임 유지하면서 안내 메시지 표시
    ret, frame = cap.read()
    if ret:
        height = frame.shape[0]
        final_y_pos = height - int(30 * font_scale)
        cv2.putText(frame, "Press 'n' for next, 'q' to quit", (10, final_y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 255), line_thickness, cv2.LINE_AA)
        cv2.imshow('Segment Replay', frame)
        
        # 키 입력 대기
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('n'): 
                break
            elif key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, segment_start + current_frame_in_segment)
        
        while True:
            if debug_mode:
                # 디버그 모드에서는 수동으로 프레임 위치 설정
                absolute_frame = segment_start + current_frame_in_segment
                if absolute_frame >= segment_end:
                    break
                cap.set(cv2.CAP_PROP_POS_FRAMES, absolute_frame)
            else:
                # 일반 모드에서는 순차 재생
                if cap.get(cv2.CAP_PROP_POS_FRAMES) >= segment_end:
                    break
            
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
                # 기존 랜드마크 표시 (어깨, 골반)
                for landmark_enum, color in LANDMARK_COLORS.items():
                    # 랜드마크의 정규화된 좌표를 실제 픽셀 좌표로 변환
                    cx = int(landmarks[landmark_enum.value].x * width)
                    cy = int(landmarks[landmark_enum.value].y * height)
                    # OpenCV를 사용하여 원(점) 그리기
                    cv2.circle(annotated_image, (cx, cy), max(5, line_thickness * 5), color, -1)
                
                # 3스텝과 4스텝에서 손목 턴아웃 분석용 랜드마크 표시
                if seg['step_num'] in [3, 4]:
                    # 턴아웃 분석 수행 (보정된 팔 사용)
                    wrist_analysis = analyze_wrist_turnout_by_ball_position(results.pose_landmarks, step_num=seg['step_num'])
                    
                    # === 보정된 결과에 따라 올바른 위치에 랜드마크 표시 ===
                    corrected_landmarks = arm_tracker.get_corrected_arm_landmarks(results.pose_landmarks, seg['step_num'])
                    correction_applied = corrected_landmarks[0] == 15  # LEFT_WRIST = 15 (보정 적용됨)
                    
                    # 디버깅 정보 출력
                    print(f"[DEBUG] corrected_landmarks[0]: {corrected_landmarks[0]}")
                    print(f"[DEBUG] correction_applied: {correction_applied}") 
                    print(f"[DEBUG] step_4_correction_needed: {getattr(arm_tracker, 'step_4_correction_needed', 'NOT_SET')}")
                    
                    # Y좌표 비교 디버깅
                    left_wrist_y = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y
                    right_wrist_y = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y
                    print(f"[DEBUG] MediaPipe LEFT_WRIST Y: {left_wrist_y:.3f}")
                    print(f"[DEBUG] MediaPipe RIGHT_WRIST Y: {right_wrist_y:.3f}")
                    print(f"[DEBUG] LEFT < RIGHT (LEFT 위쪽): {left_wrist_y < right_wrist_y}")
                    
                    if correction_applied:
                        # MediaPipe가 오인식한 경우: Y좌표 기준으로 실제 위치 파악
                        left_wrist_y = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y
                        right_wrist_y = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y
                        
                        if right_wrist_y < left_wrist_y:
                            # MediaPipe RIGHT가 실제 위쪽(실제 오른팔)
                            # 실제 오른팔 (MediaPipe RIGHT)에 R-라벨
                            actual_right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
                            actual_right_wrist_cx = int(actual_right_wrist.x * width)
                            actual_right_wrist_cy = int(actual_right_wrist.y * height)
                            cv2.circle(annotated_image, (actual_right_wrist_cx, actual_right_wrist_cy), max(4, line_thickness * 3), (0, 165, 255), -1)  # 주황색
                            cv2.putText(annotated_image, "R-Wrist", (actual_right_wrist_cx + 15, actual_right_wrist_cy), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.7, (0, 165, 255), line_thickness, cv2.LINE_AA)
                            
                            actual_right_index = landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value]
                            actual_right_index_cx = int(actual_right_index.x * width)
                            actual_right_index_cy = int(actual_right_index.y * height)
                            cv2.circle(annotated_image, (actual_right_index_cx, actual_right_index_cy), max(4, line_thickness * 3), (0, 165, 255), -1)  # 주황색
                            cv2.putText(annotated_image, "R-Index", (actual_right_index_cx + 15, actual_right_index_cy), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.7, (0, 165, 255), line_thickness, cv2.LINE_AA)
                            
                            # 실제 왼팔 (MediaPipe LEFT)에 L-라벨
                            actual_left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
                            actual_left_wrist_cx = int(actual_left_wrist.x * width)
                            actual_left_wrist_cy = int(actual_left_wrist.y * height)
                            cv2.circle(annotated_image, (actual_left_wrist_cx, actual_left_wrist_cy), max(4, line_thickness * 3), (255, 0, 0), -1)  # 파란색
                            cv2.putText(annotated_image, "L-Wrist", (actual_left_wrist_cx + 15, actual_left_wrist_cy), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.7, (255, 0, 0), line_thickness, cv2.LINE_AA)
                            
                            actual_left_index = landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value]
                            actual_left_index_cx = int(actual_left_index.x * width)
                            actual_left_index_cy = int(actual_left_index.y * height)
                            cv2.circle(annotated_image, (actual_left_index_cx, actual_left_index_cy), max(4, line_thickness * 3), (255, 0, 0), -1)  # 파란색
                            cv2.putText(annotated_image, "L-Index", (actual_left_index_cx + 15, actual_left_index_cy), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.7, (255, 0, 0), line_thickness, cv2.LINE_AA)
                        else:
                            # MediaPipe LEFT가 실제 위쪽(실제 오른팔)
                            # 실제 오른팔 (MediaPipe LEFT)에 R-라벨
                            actual_right_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
                            actual_right_wrist_cx = int(actual_right_wrist.x * width)
                            actual_right_wrist_cy = int(actual_right_wrist.y * height)
                            cv2.circle(annotated_image, (actual_right_wrist_cx, actual_right_wrist_cy), max(4, line_thickness * 3), (0, 165, 255), -1)  # 주황색
                            cv2.putText(annotated_image, "R-Wrist", (actual_right_wrist_cx + 15, actual_right_wrist_cy), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.7, (0, 165, 255), line_thickness, cv2.LINE_AA)
                            
                            actual_right_index = landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value]
                            actual_right_index_cx = int(actual_right_index.x * width)
                            actual_right_index_cy = int(actual_right_index.y * height)
                            cv2.circle(annotated_image, (actual_right_index_cx, actual_right_index_cy), max(4, line_thickness * 3), (0, 165, 255), -1)  # 주황색
                            cv2.putText(annotated_image, "R-Index", (actual_right_index_cx + 15, actual_right_index_cy), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.7, (0, 165, 255), line_thickness, cv2.LINE_AA)
                            
                            # 실제 왼팔 (MediaPipe RIGHT)에 L-라벨
                            actual_left_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
                            actual_left_wrist_cx = int(actual_left_wrist.x * width)
                            actual_left_wrist_cy = int(actual_left_wrist.y * height)
                            cv2.circle(annotated_image, (actual_left_wrist_cx, actual_left_wrist_cy), max(4, line_thickness * 3), (255, 0, 0), -1)  # 파란색
                            cv2.putText(annotated_image, "L-Wrist", (actual_left_wrist_cx + 15, actual_left_wrist_cy), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.7, (255, 0, 0), line_thickness, cv2.LINE_AA)
                            
                            actual_left_index = landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value]
                            actual_left_index_cx = int(actual_left_index.x * width)
                            actual_left_index_cy = int(actual_left_index.y * height)
                            cv2.circle(annotated_image, (actual_left_index_cx, actual_left_index_cy), max(4, line_thickness * 3), (255, 0, 0), -1)  # 파란색
                            cv2.putText(annotated_image, "L-Index", (actual_left_index_cx + 15, actual_left_index_cy), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.7, (255, 0, 0), line_thickness, cv2.LINE_AA)
                    else:
                        # MediaPipe가 올바르게 인식한 경우: MediaPipe 결과 그대로 표시
                        # 오른팔 (MediaPipe RIGHT 랜드마크)에 R- 라벨
                        right_wrist_landmark = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
                        right_wrist_cx = int(right_wrist_landmark.x * width)
                        right_wrist_cy = int(right_wrist_landmark.y * height)
                        cv2.circle(annotated_image, (right_wrist_cx, right_wrist_cy), max(4, line_thickness * 3), (0, 165, 255), -1)  # 주황색
                        cv2.putText(annotated_image, "R-Wrist", (right_wrist_cx + 15, right_wrist_cy), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.7, (0, 165, 255), line_thickness, cv2.LINE_AA)
                        
                        right_index_landmark = landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value]
                        right_index_cx = int(right_index_landmark.x * width)
                        right_index_cy = int(right_index_landmark.y * height)
                        cv2.circle(annotated_image, (right_index_cx, right_index_cy), max(4, line_thickness * 3), (0, 165, 255), -1)  # 주황색
                        cv2.putText(annotated_image, "R-Index", (right_index_cx + 15, right_index_cy), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.7, (0, 165, 255), line_thickness, cv2.LINE_AA)
                        
                        # 왼팔 (MediaPipe LEFT 랜드마크)에 L- 라벨
                        left_wrist_landmark = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
                        left_wrist_cx = int(left_wrist_landmark.x * width)
                        left_wrist_cy = int(left_wrist_landmark.y * height)
                        cv2.circle(annotated_image, (left_wrist_cx, left_wrist_cy), max(4, line_thickness * 3), (255, 0, 0), -1)  # 파란색
                        cv2.putText(annotated_image, "L-Wrist", (left_wrist_cx + 15, left_wrist_cy), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.7, (255, 0, 0), line_thickness, cv2.LINE_AA)
                        
                        left_index_landmark = landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value]
                        left_index_cx = int(left_index_landmark.x * width)
                        left_index_cy = int(left_index_landmark.y * height)
                        cv2.circle(annotated_image, (left_index_cx, left_index_cy), max(4, line_thickness * 3), (255, 0, 0), -1)  # 파란색
                        cv2.putText(annotated_image, "L-Index", (left_index_cx + 15, left_index_cy), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.7, (255, 0, 0), line_thickness, cv2.LINE_AA)
                    
                    # === 개선된 공 위치 계산 - 초록색 (보정된 팔 사용) ===
                    ball_x, ball_y = detect_ball_position(results.pose_landmarks, step_num=seg['step_num'])
                    if ball_x is not None and ball_y is not None:
                        ball_cx = int(ball_x * width)
                        ball_cy = int(ball_y * height)
                        cv2.circle(annotated_image, (ball_cx, ball_cy), max(4, line_thickness * 3), (0, 255, 0), -1)
                        cv2.putText(annotated_image, "Ball", (ball_cx + 15, ball_cy), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.7, (0, 255, 0), line_thickness, cv2.LINE_AA)
                        
                        # 수직 기준선 그리기 (공의 x축에서 수직선)
                        cv2.line(annotated_image, (ball_cx, 0), (ball_cx, height), (0, 255, 255), 2)
                    
                    
                    # 턴아웃 상태 텍스트 표시
                    turnout_status = "Turn-out!" if wrist_analysis.get("turnout_detected", False) else "Normal"
                    status_color = (0, 0, 255) if wrist_analysis.get("turnout_detected", False) else (0, 255, 0)
                    cv2.putText(annotated_image, f"Wrist: {turnout_status}", (10, height - 50), cv2.FONT_HERSHEY_SIMPLEX, font_scale, status_color, line_thickness, cv2.LINE_AA)

            y_offset = int(30 * font_scale)
            
            # 디버그 모드일 때 프레임 정보 추가 표시
            if debug_mode:
                absolute_frame = segment_start + current_frame_in_segment
                debug_info = f"{seg['label']} | Frame: {absolute_frame}/{total_frames-1} ({absolute_frame/fps:.2f}s)"
                cv2.putText(annotated_image, debug_info, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, (255, 255, 0), line_thickness, cv2.LINE_AA)
                y_offset += int(35 * font_scale)
                
                segment_info = f"Segment {current_segment_idx + 1}/{len(segments)} | Frame {current_frame_in_segment + 1}/{segment_length}"
                cv2.putText(annotated_image, segment_info, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.6, (200, 200, 200), line_thickness, cv2.LINE_AA)
                y_offset += int(30 * font_scale)
            else:
                cv2.putText(annotated_image, seg['label'], (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 0), line_thickness, cv2.LINE_AA)
                y_offset += int(40 * font_scale)
            
            step_angle = angles.get(seg['step_num'])
            if step_angle is not None:
                angle_text = f"Torso Angle: {step_angle:.2f} deg"
                cv2.putText(annotated_image, angle_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), line_thickness, cv2.LINE_AA)
                y_offset += int(40 * font_scale)


            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # 디버그 모드일 때 손목/검지 좌표 상세 표시
                if debug_mode:
                    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
                    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value] 
                    left_index = landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value]
                    right_index = landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value]
                    
                    # 볼 위치 계산
                    ball_x, ball_y = detect_ball_position(results.pose_landmarks, step_num=seg['step_num'])
                    
                    coord_texts = [
                        f"L_Wrist: ({left_wrist.x:.3f}, {left_wrist.y:.3f}) vis:{left_wrist.visibility:.2f}",
                        f"R_Wrist: ({right_wrist.x:.3f}, {right_wrist.y:.3f}) vis:{right_wrist.visibility:.2f}", 
                        f"L_Index: ({left_index.x:.3f}, {left_index.y:.3f}) vis:{left_index.visibility:.2f}",
                        f"R_Index: ({right_index.x:.3f}, {right_index.y:.3f}) vis:{right_index.visibility:.2f}",
                        f"Ball: ({ball_x:.3f}, {ball_y:.3f})" if ball_x is not None else "Ball: (None, None)"
                    ]
                    
                    for coord_text in coord_texts:
                        cv2.putText(annotated_image, coord_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.5, (255, 255, 255), line_thickness, cv2.LINE_AA)
                        y_offset += int(25 * font_scale)
                else:
                    # 일반 모드에서는 기존 visibility 표시
                    left_ankle_vis = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].visibility
                    right_ankle_vis = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].visibility
                    vis_text = f"L-Ankle Vis: {left_ankle_vis:.2f} | R-Ankle Vis: {right_ankle_vis:.2f}"
                    cv2.putText(annotated_image, vis_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, (255, 255, 255), line_thickness, cv2.LINE_AA)
                    y_offset += int(30 * font_scale)

            current_step_feedback = all_analysis.get('torso',{}).get('feedback',{}).get(seg['step_num'], []) + \
                                    all_analysis.get('foot',{}).get(seg['step_num'], [])
            
            # 3스텝과 4스텝에서 손목 피드백 추가
            if seg['step_num'] in [3, 4]:
                current_step_feedback.extend(all_analysis.get('wrist',{}).get('feedback',{}).get(seg['step_num'], []))
            
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
            
            # 디버그 모드일 때 하단에 조작 안내 추가
            if debug_mode:
                controls_y = height - 60
                cv2.putText(annotated_image, "←/→(A/D):1frame ↑/↓(W/S):10frames SPACE:play/pause N:next Q:quit", 
                           (10, controls_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.5, (0, 255, 255), line_thickness, cv2.LINE_AA)
                
                # 자동재생 상태 표시
                play_status = "AUTO-PLAY" if auto_play else "PAUSED"
                status_color = (0, 255, 0) if auto_play else (0, 0, 255)
                cv2.putText(annotated_image, f"[{play_status}]", (width - 120, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.6, status_color, line_thickness, cv2.LINE_AA)

            last_frame_in_segment = annotated_image.copy()
            cv2.imshow('Segment Replay', annotated_image)

            # 키 입력 처리 - 디버그 모드와 일반 모드 구분
            if debug_mode:
                # 디버그 모드: 키 입력 대기하면서 프레임 제어
                key = cv2.waitKey(0) & 0xFF
                
                if key == ord('q') or key == 27:  # 'q' or ESC
                    cap.release() 
                    cv2.destroyAllWindows()
                    return
                elif key == 83 or key == ord('d'):  # Right arrow or 'd' - 1프레임 앞으로
                    current_frame_in_segment = min(current_frame_in_segment + 1, segment_length - 1)
                elif key == 81 or key == ord('a'):  # Left arrow or 'a' - 1프레임 뒤로  
                    current_frame_in_segment = max(current_frame_in_segment - 1, 0)
                elif key == 82 or key == ord('w'):  # Up arrow or 'w' - 10프레임 앞으로
                    current_frame_in_segment = min(current_frame_in_segment + 10, segment_length - 1)
                elif key == 84 or key == ord('s'):  # Down arrow or 's' - 10프레임 뒤로
                    current_frame_in_segment = max(current_frame_in_segment - 10, 0)
                elif key == ord('n'):  # 'n' - 다음 세그먼트로
                    break
                elif key == ord(' '):  # SPACE - 자동 재생 토글
                    auto_play = not auto_play
                    if auto_play:
                        print("Auto-play enabled")
                    else:
                        print("Auto-play disabled")
                
                # 자동 재생 모드일 때
                if auto_play:
                    current_frame_in_segment = min(current_frame_in_segment + 1, segment_length - 1)
                    if current_frame_in_segment >= segment_length - 1:
                        break  # 세그먼트 끝에 도달하면 다음 세그먼트로
            else:
                # 일반 모드: 기존 로직 유지
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return
                
                # 일반 모드에서는 자동으로 다음 프레임으로
        
        if last_frame_in_segment is not None:
            # 마지막 프레임에 안내 텍스트 추가
            final_y_pos = last_frame_in_segment.shape[0] - int(30 * font_scale)
            cv2.putText(last_frame_in_segment, "Press 'n' for next, 'q' to quit", (10, final_y_pos), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 255), line_thickness, cv2.LINE_AA)
            cv2.imshow('Segment Replay', last_frame_in_segment)
            
            # 대기 시간 없이 바로 키 입력 대기
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

    marked_steps = user_driven_step_segmentation(video_file)
    
    if marked_steps and len(marked_steps) == 5:
        all_analysis_results = {
            "torso": analyze_torso_angle(marked_steps),
            "foot": analyze_foot_crossover_by_x(marked_steps),
            "wrist": analyze_wrist_turnout(marked_steps),
        }
        
        print("\nReplaying marked segments with analysis data...")
        print("Press 'y' for DEBUG mode (frame-by-frame navigation) or any other key for normal mode:")
        
        # 사용자 입력 받기
        import sys
        if sys.platform == "win32":
            import msvcrt
            user_choice = msvcrt.getch().decode('utf-8').lower()
        else:
            user_choice = input().lower()
        
        debug_mode = user_choice == 'y'
        
        if debug_mode:
            print("DEBUG MODE activated - Use arrow keys to navigate frames!")
            # 디버그 콜백 함수들 전달
            replay_step_segments(video_file, marked_steps, all_analysis_results, 
                                ui_renderer=debug_ui_renderer,
                                landmark_renderer=debug_landmark_renderer,
                                input_handler=debug_input_handler)
        else:
            print("Normal replay mode")
            # 일반 모드는 콜백 없이 호출
            replay_step_segments(video_file, marked_steps, all_analysis_results)
        
        print("\n--- Final Posture Analysis Summary ---")
        torso_feedback = all_analysis_results['torso'].get('feedback', {})
        foot_feedback = all_analysis_results['foot']
        wrist_feedback = all_analysis_results['wrist'].get('feedback', {})
        
        all_feedback_merged = {**torso_feedback, **foot_feedback, **wrist_feedback}
        for step in sorted(all_feedback_merged.keys()):
            feedback_lists = [
                torso_feedback.get(step, []),
                foot_feedback.get(step, []),
            ]
            # 3스텝과 4스텝에서 손목 피드백 포함
            if step in [3, 4]:
                feedback_lists.append(wrist_feedback.get(step, []))
                
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