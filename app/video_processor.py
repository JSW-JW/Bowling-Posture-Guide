# app/video_processor.py
import cv2

def load_video(video_path):
    """
    지정된 경로의 영상을 로드합니다.
    
    :param video_path: 영상 파일 경로
    :return: cv2.VideoCapture 객체
    :raises FileNotFoundError: 파일이 없거나 열리지 않을 경우
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"비디오 파일을 열 수 없습니다: {video_path}")
    return cap

def extract_frames(video_path, every_n_frames=1):
    """
    영상을 프레임 단위로 추출합니다. every_n_frames 값에 따라 간격을 지정할 수 있습니다.
    
    :param video_path: 영상 파일 경로
    :param every_n_frames: 몇 프레임마다 프레임을 추출할지 (기본값 1, 즉 모든 프레임)
    :return: 추출된 프레임 리스트 (BGR 형식)
    """
    cap = load_video(video_path)
    frames = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % every_n_frames == 0:
            frames.append(frame)
        frame_count += 1

    cap.release()
    return frames

if __name__ == "__main__":
    # 간단한 테스트: 영상 파일을 로드하여 프레임 수 출력
    video_path = "static/sample_video.mp4"
    try:
        frames = extract_frames(video_path, every_n_frames=10)
        print(f"{video_path}에서 {len(frames)}개의 프레임을 추출했습니다.")
    except FileNotFoundError as e:
        print(e)
