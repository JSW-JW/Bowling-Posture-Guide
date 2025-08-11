from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime
from enum import Enum

class AnalysisFeedback(BaseModel):
    torso: Dict[int, List[str]]
    foot: Dict[int, List[str]]
    stability: Dict[int, List[str]]

class AnalysisResult(BaseModel):
    feedback: AnalysisFeedback
    visualizations: Dict[str, str]

class RoomType(str, Enum):
    FEEDBACK = "feedback"
    GENERAL = "general"

class Room(BaseModel):
    id: str
    name: str
    room_type: RoomType
    description: Optional[str] = None
    max_users: Optional[int] = None
    created_at: datetime
    created_by: Optional[str] = None  # Future: user_id for authentication
    is_active: bool = True
    
class RoomCreateRequest(BaseModel):
    name: str
    room_type: RoomType = RoomType.FEEDBACK
    description: Optional[str] = None
    max_users: Optional[int] = 10

class RoomJoinRequest(BaseModel):
    username: Optional[str] = None

class ChatMessage(BaseModel):
    id: str
    room_id: str
    client_id: str
    username: str
    message: str
    timestamp: datetime
    message_type: str = "message"  # "message", "join", "leave"

class ChatResponse(BaseModel):
    type: str  # "message", "user_joined", "user_left", "user_list", "room_info"
    data: dict

class RoomInfo(BaseModel):
    room: Room
    user_count: int
    users: List[Dict[str, str]]
