#!/usr/bin/env python3
"""
Step Completion Detection - Version 3.2 (Image Result Export)
Detects step completions, and saves the result frames as images.
"""

import cv2
import numpy as np
import mediapipe as mp
import json
import os
from typing import List, Dict, Tuple, Any

# MediaPipe Pose 솔루션 초기화
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

class PoseAnalyzer:
    """MediaPipe를 사용해 영상에서 포즈 랜드마크를 추출하는 클래스"""

    def __init__(self, static_image_mode=False, model_complexity=1, min_detection_confidence=0.5):
        self.pose = mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence
        )

    def get_landmarks_from_video(self, video_path: str) -> Tuple[List[Dict], Dict]:
        """비디오 파일에서 모든 프레임의 랜드마크를 추출합니다."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {video_path}")

        video_meta = {
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        }
        
        all_landmarks = []
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(image_rgb)
            
            frame_landmarks = {'frame': frame_idx, 'landmarks_mp': None, 'landmarks_dict': None}
            if results.pose_landmarks:
                frame_landmarks['landmarks_mp'] = results.pose_landmarks
                
                landmarks_dict = {}
                for idx, lm in enumerate(results.pose_landmarks.landmark):
                    landmark_name = mp_pose.PoseLandmark(idx).name
                    landmarks_dict[landmark_name] = {'x': lm.x, 'y': lm.y, 'z': lm.z}
                frame_landmarks['landmarks_dict'] = landmarks_dict
                
            all_landmarks.append(frame_landmarks)
            frame_idx += 1
            
        cap.release()
        self.pose.close()
        return all_landmarks, video_meta

class StepDetector:
    """랜드마크 데이터를 기반으로 스텝 완료 지점을 감지하는 클래스"""
    
    def __init__(self,
                 smoothing_window: int = 5,
                 velocity_threshold: float = 5.0,
                 min_step_interval_frames: int = 10):
        self.smoothing_window = smoothing_window
        self.velocity_threshold = velocity_threshold
        self.min_step_interval_frames = min_step_interval_frames

    def _calculate_velocities(self, all_landmarks: List[Dict], width: int, height: int) -> Dict[str, List[float]]:
        velocities = {'left': [], 'right': []}
        for i in range(1, len(all_landmarks)):
            prev_lms = all_landmarks[i-1]['landmarks_dict']
            curr_lms = all_landmarks[i]['landmarks_dict']
            
            if prev_lms and curr_lms and 'LEFT_ANKLE' in curr_lms and 'RIGHT_ANKLE' in curr_lms:
                prev_left = prev_lms['LEFT_ANKLE']
                curr_left = curr_lms['LEFT_ANKLE']
                dist_left = np.sqrt(((curr_left['x'] - prev_left['x']) * width)**2 + 
                                    ((curr_left['y'] - prev_left['y']) * height)**2)
                
                prev_right = prev_lms['RIGHT_ANKLE']
                curr_right = curr_lms['RIGHT_ANKLE']
                dist_right = np.sqrt(((curr_right['x'] - prev_right['x']) * width)**2 + 
                                     ((curr_right['y'] - prev_right['y']) * height)**2)
                
                velocities['left'].append(dist_left)
                velocities['right'].append(dist_right)
            else:
                velocities['left'].append(0)
                velocities['right'].append(0)
        return velocities

    def _smooth_velocity(self, velocity_data: List[float]) -> List[float]:
        if not velocity_data: return []
        kernel = np.ones(self.smoothing_window) / self.smoothing_window
        return np.convolve(velocity_data, kernel, mode='same')

    # StepDetector 클래스 안에 있는 detect_steps 함수만 수정합니다.

    def detect_steps(self, all_landmarks: List[Dict], video_meta: Dict) -> List[Dict]:
        # --- 1단계: 속도 기반으로 전체 스텝을 우선 감지 ---
        # (이 부분은 이전 코드와 완전히 동일합니다)
        width, height = video_meta['width'], video_meta['height']
        velocities = self._calculate_velocities(all_landmarks, width, height)
        smooth_vel_left = self._smooth_velocity(velocities['left'])
        smooth_vel_right = self._smooth_velocity(velocities['right'])
        
        if len(smooth_vel_left) == 0: return []

        initial_completions = []
        step_number = 1
        last_completion_frame = -self.min_step_interval_frames
        left_foot_state, right_foot_state = 'STOPPED', 'STOPPED'
        step_sequence = ['left', 'right', 'left', 'right', 'left']

        for i in range(len(smooth_vel_left)):
            frame_idx = i + 1
            current_right_state = 'MOVING' if smooth_vel_right[i] > self.velocity_threshold else 'STOPPED'
            current_left_state = 'MOVING' if smooth_vel_left[i] > self.velocity_threshold else 'STOPPED'

            if step_number > len(step_sequence): break
            target_foot = step_sequence[step_number - 1]

            foot_state = current_left_state if target_foot == 'left' else current_right_state
            prev_foot_state = left_foot_state if target_foot == 'left' else right_foot_state

            if prev_foot_state == 'MOVING' and foot_state == 'STOPPED' and \
            (frame_idx - last_completion_frame) > self.min_step_interval_frames:
                
                initial_completions.append({
                    'step_number': step_number, 'foot': target_foot,
                    'completion_frame': frame_idx,
                    'completion_timestamp': frame_idx / video_meta['fps']
                })
                last_completion_frame = frame_idx
                step_number += 1
            
            right_foot_state, left_foot_state = current_right_state, current_left_state
        
        # --- 2단계: 3스텝 재탐색 및 보정 로직 (멘티님 최종 아이디어 적용) ---
        print("\n🔄 Re-searching for Step 3 using constrained heuristic (Y-coord + Stability)...")
        steps_dict = {step['step_number']: step for step in initial_completions}

        if 2 in steps_dict and 4 in steps_dict:
            step2_frame = steps_dict[2]['completion_frame']
            step4_frame = steps_dict[4]['completion_frame']

            # ❗️❗️❗️ 수정된 재탐색 로직 시작 ❗️❗️❗️
            
            # 기준점 설정: 2스텝 완료 시점의 오른발 y좌표
            step2_landmarks = all_landmarks[step2_frame].get('landmarks_dict')
            if not (step2_landmarks and 'RIGHT_ANKLE' in step2_landmarks):
                print("⚠️ Cannot find right ankle at Step 2 to set anchor. Aborting heuristic search.")
                return initial_completions # 문제가 있으면 그냥 초기 감지 결과 반환

            right_foot_y_anchor = step2_landmarks['RIGHT_ANKLE']['y']

            # 1. 1차 필터링: 왼발이 높은 프레임 후보군 찾기
            candidates = []
            for frame_idx in range(step2_frame, step4_frame):
                landmarks = all_landmarks[frame_idx].get('landmarks_dict')
                if landmarks and 'LEFT_ANKLE' in landmarks:
                    candidates.append((landmarks['LEFT_ANKLE']['y'], frame_idx))
            
            if not candidates:
                print("⚠️ No valid frames found in the search window for Step 3.")
                return initial_completions

            # y좌표가 낮은 순(높이 올라간 순)으로 정렬 후 상위 10개 후보 선정
            candidates.sort(key=lambda x: x[0])
            top_n = 10
            top_candidates = candidates[:top_n]

            # 2. 2차 최종 선택: 후보군 중에서 오른발이 가장 안정적인 프레임 선택
            min_stability_error = float('inf')
            best_frame_for_step3 = -1

            for _, frame_idx in top_candidates:
                landmarks = all_landmarks[frame_idx].get('landmarks_dict')
                if landmarks and 'RIGHT_ANKLE' in landmarks:
                    current_right_foot_y = landmarks['RIGHT_ANKLE']['y']
                    
                    # 안정성 오차 계산: 2스텝 기준점과의 y좌표 차이
                    stability_error = abs(current_right_foot_y - right_foot_y_anchor)

                    if stability_error < min_stability_error:
                        min_stability_error = stability_error
                        best_frame_for_step3 = frame_idx
            # ❗️❗️❗️ 수정된 재탐색 로직 끝 ❗️❗️❗️
            
            if best_frame_for_step3 != -1:
                print(f"✅ Heuristic search (Y-coord) found a better frame for Step 3: {best_frame_for_step3}")
                steps_dict[3] = {
                    'step_number': 3, 'foot': 'left',
                    'completion_frame': best_frame_for_step3,
                    'completion_timestamp': best_frame_for_step3 / video_meta['fps']
                }
            else:
                print("⚠️ Heuristic search (Y-coord) could not find a suitable frame for Step 3.")
        else:
            print("ℹ️ Step 2 and 4 not available for heuristic search.")

        final_completions = sorted(list(steps_dict.values()), key=lambda x: x['step_number'])
        return final_completions



def save_result_images(video_path: str, detected_steps: List[Dict], all_landmarks: List[Dict], output_dir: str):
    """감지된 스텝의 프레임을 이미지 파일로 저장합니다."""
    print(f"\n🖼️ Saving result images to '{output_dir}'...")
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Could not open video to save images.")
        return

    # 빠른 조회를 위해 랜드마크 데이터를 딕셔너리로 변환
    landmark_map = {lm['frame']: lm.get('landmarks_mp') for lm in all_landmarks}

    for step in detected_steps:
        frame_idx = step['completion_frame']
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        
        ret, frame = cap.read()
        if ret:
            # 랜드마크 그리기
            landmarks_to_draw = landmark_map.get(frame_idx)
            if landmarks_to_draw:
                mp_drawing.draw_landmarks(
                    frame,
                    landmarks_to_draw,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                )

            # 텍스트 추가
            text = f"Step {step['step_number']} ({step['foot']}) Completion"
            cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # 파일로 저장
            img_path = os.path.join(output_dir, f"step_{step['step_number']}_completion.png")
            cv2.imwrite(img_path, frame)
            print(f"  - Saved '{img_path}'")

    cap.release()

if __name__ == "__main__":
    # ❗️❗️❗️ 본인 영상 경로에 맞게 수정해주세요 ❗️❗️❗️
    video_path = "신승현프로_2.mp4" 
    
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
    else:
        print("🚀 Starting Bowling Pose Analysis...")
        
        # 1. 랜드마크 추출
        pose_analyzer = PoseAnalyzer()
        all_landmarks, video_meta = pose_analyzer.get_landmarks_from_video(video_path)
        print(f"✅ Landmark extraction complete. Total frames: {len(all_landmarks)}")

        # 2. StepDetector '객체' 생성 (자동차 만들기)
        step_detector = StepDetector(
            smoothing_window=5,
            velocity_threshold=5.0,
            min_step_interval_frames=10
        )
        
        # 3. 스텝 감지 실행
        detected_steps = step_detector.detect_steps(all_landmarks, video_meta)

        # 4. 결과 텍스트 출력
        print("\n📊 Final Step Detection Results:")
        for step in detected_steps:
            print(f"  - Step {step['step_number']} ({step['foot']}): "
                  f"Frame {step['completion_frame']} at {step['completion_timestamp']:.3f}s")
                  
        # 5. 결과 JSON 저장
        # (이 부분은 필요에 따라 순서를 바꿔도 됩니다)
        output_path = "step_detection_results.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(detected_steps, f, indent=2)
        print(f"\n💾 Results saved to: {output_path}")

        # 6. 속도 그래프 그려보기 (디버깅용) - step_detector 객체가 생성된 후 실행
        import matplotlib.pyplot as plt

        velocities = step_detector._calculate_velocities(all_landmarks, video_meta['width'], video_meta['height'])
        smooth_vel_left = step_detector._smooth_velocity(velocities['left'])
        smooth_vel_right = step_detector._smooth_velocity(velocities['right'])
        
        plt.figure(figsize=(15, 5))
        plt.plot(smooth_vel_left, label='Left Foot Velocity', color='blue')
        plt.plot(smooth_vel_right, label='Right Foot Velocity', color='red', linestyle='--')
        plt.axhline(y=step_detector.velocity_threshold, color='green', linestyle=':', label='Velocity Threshold')
        
        for step in detected_steps:
            color = 'blue' if step['foot'] == 'left' else 'red'
            # label이 중복해서 표시되지 않도록 수정
            step_label = f"Step {step['step_number']} ({step['foot']})"
            if step_label not in plt.gca().get_legend_handles_labels()[1]:
                 plt.axvline(x=step['completion_frame'], color=color, linestyle='-', label=step_label)
            else:
                 plt.axvline(x=step['completion_frame'], color=color, linestyle='-')

        plt.title('Foot Velocity Over Time')
        plt.xlabel('Frame')
        plt.ylabel('Smoothed Velocity')
        plt.legend()
        plt.grid(True)
        plt.savefig('velocity_graph.png')
        print("\n📈 Velocity graph saved to 'velocity_graph.png'")

        # 7. 결과 이미지 저장
        if detected_steps:
            save_result_images(
                video_path=video_path,
                detected_steps=detected_steps,
                all_landmarks=all_landmarks,
                output_dir="step_images"
            )
        
        print("\n🎉 Analysis finished!")