from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import shutil
import os
from collections import defaultdict

from models import AnalysisResult, AnalysisFeedback
import services

app = FastAPI()

# --- CORS 설정 ---
origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 임시 파일 저장을 위한 디렉토리 ---
TEMP_DIR = "temp_videos"
os.makedirs(TEMP_DIR, exist_ok=True)


@app.post("/analyze/interactive-steps", response_model=AnalysisResult)
async def analyze_interactive_steps(
    video: UploadFile = File(...), 
    timestamps: List[float] = Form(...)
):
    """
    동영상 파일과 타임스탬프를 받아 자세를 분석하고 결과를 반환합니다.
    """
    temp_video_path = os.path.join(TEMP_DIR, video.filename)
    try:
        with open(temp_video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

        # 1. 랜드마크 데이터 추출
        marked_steps = services.extract_landmarks_from_timestamps(temp_video_path, timestamps)
        if not marked_steps or len(marked_steps) < 5:
            raise HTTPException(status_code=400, detail="Failed to detect pose in one or more steps.")

        # 2. 모든 분석 수행
        torso_analysis = services.analyze_torso_angle(marked_steps)
        foot_analysis = services.analyze_foot_crossover_by_x(marked_steps)
        # stability_analysis = services.analyze_sliding_stability(marked_steps) # 추후 추가 가능

        # 3. 분석 결과 시각화
        visualizations = services.visualize_analysis(temp_video_path, marked_steps, torso_analysis)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)

    # 4. 피드백 데이터 재구성
    # defaultdict를 사용하여 스텝별로 모든 피드백을 통합
    merged_feedback = defaultdict(list)
    for step, feedback_list in torso_analysis.get("feedback", {}).items():
        merged_feedback[step].extend(feedback_list)
    for step, feedback_list in foot_analysis.items():
        merged_feedback[step].extend(feedback_list)
    # for step, feedback_list in stability_analysis.items():
    #     merged_feedback[step].extend(feedback_list)

    # Pydantic 모델에 맞게 최종 페이로드 생성
    feedback_payload = {
        "torso": {k: v for k, v in merged_feedback.items() if any("[Torso]" in s for s in v)},
        "foot": {k: v for k, v in merged_feedback.items() if any("[Foot]" in s for s in v)},
        "stability": {} # 아직 안정성 분석이 없으므로 비워둠
    }

    return AnalysisResult(
        feedback=AnalysisFeedback(**feedback_payload),
        visualizations=visualizations
    )

@app.get("/")
def read_root():
    return {"message": "Bowling Posture Guide API is running."}
