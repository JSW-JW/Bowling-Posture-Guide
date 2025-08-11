from fastapi import FastAPI, File, UploadFile, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import shutil
import os
import asyncio
import json
import uuid
from datetime import datetime
from contextlib import asynccontextmanager

from models import AnalysisResult, AnalysisFeedback, Room, RoomCreateRequest, RoomJoinRequest, RoomInfo, RoomType
import services
from websocket_manager import manager, pubsub_manager
from room_manager import room_manager

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

# Old WebSocket endpoint removed - now using room-based endpoint only

# --- Room Management API ---
@app.post("/rooms", response_model=Room)
async def create_room(request: RoomCreateRequest):
    """Create a new feedback room"""
    room = room_manager.create_room(request)
    return room

@app.get("/rooms", response_model=List[Room])
async def list_rooms(room_type: Optional[RoomType] = None):
    """List all active rooms"""
    rooms = room_manager.list_rooms(room_type=room_type)
    return rooms

@app.get("/rooms/{room_id}", response_model=RoomInfo)
async def get_room_info(room_id: str):
    """Get detailed room information"""
    room = room_manager.get_room(room_id)
    if not room:
        raise HTTPException(status_code=404, detail="Room not found")
    
    users = manager.get_room_users(room_id)
    
    return RoomInfo(
        room=room,
        user_count=len(users),
        users=users
    )

@app.post("/rooms/{room_id}/join")
async def join_room(room_id: str, request: RoomJoinRequest):
    """Join a room (Future: require authentication)"""
    room = room_manager.get_room(room_id)
    if not room:
        raise HTTPException(status_code=404, detail="Room not found")
    
    if not room.is_active:
        raise HTTPException(status_code=400, detail="Room is not active")
    
    # Future: Extract client_id from JWT token
    # For now, return room info for client to connect via WebSocket
    users = manager.get_room_users(room_id)
    
    return {
        "success": True,
        "room": room,
        "user_count": len(users),
        "websocket_url": f"/ws/{room_id}/{{client_id}}"
    }

@app.websocket("/ws/{room_id}/{client_id}")
async def websocket_room_endpoint(websocket: WebSocket, room_id: str, client_id: str, username: str = None):
    """WebSocket endpoint for room-based chat"""
    # Validate room exists
    room = room_manager.get_room(room_id)
    if not room or not room.is_active:
        await websocket.close(code=4004, reason="Room not found or inactive")
        return
    
    username = username or f"User_{client_id[:8]}"
    
    # Join room in room manager
    if not room_manager.join_room(client_id, room_id):
        await websocket.close(code=4003, reason="Cannot join room (may be full)")
        return
    
    # Connect to WebSocket manager
    await manager.connect(websocket, client_id, username, room_id)
    
    # Subscribe to room's Redis channel
    await pubsub_manager.subscribe_to_room(room_id)
    
    # Send room info to the new user
    users = manager.get_room_users(room_id)
    room_dict = room.dict()
    room_dict['created_at'] = room.created_at.isoformat()  # Convert datetime to string
    await manager.send_personal_message({
        "type": "room_info",
        "data": {
            "room": room_dict,
            "users": users,
            "user_count": len(users)
        }
    }, client_id)
    
    # Broadcast user joined message
    await pubsub_manager.publish_to_room({
        "type": "user_joined",
        "data": {
            "client_id": client_id,
            "username": username,
            "message": f"{username}님이 {room.name}에 입장했습니다"
        }
    }, room_id)
    
    # Broadcast updated user list to room
    await pubsub_manager.publish_to_room({
        "type": "user_list",
        "data": {"users": manager.get_room_users(room_id)}
    }, room_id)
    
    try:
        while True:
            data = await websocket.receive_text()
            try:
                message_data = json.loads(data)
                
                if message_data.get("type") == "chat_message":
                    chat_message = {
                        "type": "message",
                        "data": {
                            "id": str(uuid.uuid4()),
                            "room_id": room_id,
                            "client_id": client_id,
                            "username": manager.get_username(client_id),
                            "message": message_data.get("message", ""),
                            "timestamp": datetime.now().isoformat()
                        }
                    }
                    await pubsub_manager.publish_to_room(chat_message, room_id)
                    
            except json.JSONDecodeError:
                # Fallback for plain text messages
                chat_message = {
                    "type": "message",
                    "data": {
                        "id": str(uuid.uuid4()),
                        "room_id": room_id,
                        "client_id": client_id,
                        "username": manager.get_username(client_id),
                        "message": data,
                        "timestamp": datetime.now().isoformat()
                    }
                }
                await pubsub_manager.publish_to_room(chat_message, room_id)
                
    except WebSocketDisconnect:
        username = manager.get_username(client_id)
        
        # Disconnect from managers
        manager.disconnect(client_id)
        room_manager.leave_room(client_id)
        
        # Broadcast user left message
        await pubsub_manager.publish_to_room({
            "type": "user_left",
            "data": {
                "client_id": client_id,
                "username": username,
                "message": f"{username}님이 {room.name}에서 나갔습니다"
            }
        }, room_id)
        
        # Broadcast updated user list to room
        await pubsub_manager.publish_to_room({
            "type": "user_list",
            "data": {"users": manager.get_room_users(room_id)}
        }, room_id)

@app.get("/")
def read_root():
    return {"message": "Bowling Posture Guide API is running."}