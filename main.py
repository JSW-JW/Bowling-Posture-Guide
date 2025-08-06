from fastapi import FastAPI, File, UploadFile, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import shutil
import os
import asyncio
from contextlib import asynccontextmanager

from models import AnalysisResult, AnalysisFeedback
import services
from websocket_manager import manager, pubsub_manager

# --- FastAPI 생명주기 이벤트 ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 서버 시작 시
    await pubsub_manager.connect_to_redis()
    await pubsub_manager.subscribe()
    # Redis 리스너를 백그라운드 태스크로 실행
    redis_listener_task = asyncio.create_task(pubsub_manager.listen())
    yield
    # 서버 종료 시
    print("Shutting down...")
    redis_listener_task.cancel()
    await pubsub_manager.close()

app = FastAPI(lifespan=lifespan)

# --- CORS 설정 ---
origins = ["http://localhost", "http://localhost:3000"]
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
    """동영상과 타임스탬프를 받아 자세를 분석하고 결과를 반환합니다."""
    temp_video_path = os.path.join(TEMP_DIR, video.filename)
    try:
        with open(temp_video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

        marked_steps = services.extract_landmarks_from_timestamps(temp_video_path, timestamps)
        if not marked_steps or len(marked_steps) < 5:
            raise HTTPException(status_code=400, detail="Failed to detect pose in one or more steps.")

        torso_analysis = services.analyze_torso_angle(marked_steps)
        foot_analysis = services.analyze_foot_crossover_by_x(marked_steps)
        stability_analysis = services.analyze_sliding_stability(marked_steps)
        visualizations = services.visualize_analysis(temp_video_path, marked_steps, torso_analysis)

    except Exception as e:
        print(f"An error occurred during analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)

    # Pydantic 모델이 기대하는 형식에 정확히 맞춰서 페이로드 구성
    feedback_payload = {
        "torso": torso_analysis.get("feedback", {}),
        "foot": foot_analysis,
        "stability": stability_analysis
    }
    
    return AnalysisResult(
        feedback=AnalysisFeedback(**feedback_payload),
        visualizations=visualizations
    )

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    try:
        while True:
            data = await websocket.receive_text()
            await pubsub_manager.publish(f"Client #{client_id} says: {data}")
    except WebSocketDisconnect:
        manager.disconnect(client_id)
        await pubsub_manager.publish(f"Client #{client_id} left the chat")

@app.get("/")
def read_root():
    return {"message": "Bowling Posture Guide API is running."}