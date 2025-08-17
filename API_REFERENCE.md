# 🔌 API Reference - Bowling Posture Guide

## Overview

The Bowling Posture Guide API is built with FastAPI and provides endpoints for video analysis, room management, and real-time communication. All endpoints return JSON responses and follow RESTful conventions.

**Base URL**: `http://localhost:8000`  
**API Documentation**: `http://localhost:8000/docs` (Swagger UI)  
**API Schema**: `http://localhost:8000/redoc` (ReDoc)

---

## 🎥 Video Analysis API

### Analyze Interactive Steps

Analyzes a bowling video with user-provided timestamps for each of the 5 steps.

```http
POST /analyze/interactive-steps
```

#### Request

**Content-Type**: `multipart/form-data`

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `video` | File | ✅ | Bowling video file (MP4 recommended) |
| `timestamps` | Array<float> | ✅ | Timestamps for steps 1-5 in seconds |

**Example Request:**
```javascript
const formData = new FormData();
formData.append('video', videoFile);
formData.append('timestamps', JSON.stringify([1.0, 2.5, 4.0, 5.5, 7.0]));

fetch('/analyze/interactive-steps', {
  method: 'POST',
  body: formData
});
```

#### Response

**Status**: `200 OK`  
**Content-Type**: `application/json`

```json
{
  "feedback": {
    "torso": {
      "2": ["[Torso] Good: Proper stance"],
      "3": ["[Torso] Good: Proper tilt"],
      "4": ["[Torso] Good: Proper tilt"],
      "5": ["[Torso] Good: Angle maintained"]
    },
    "foot": {
      "2": ["[Foot] Good: Crossover is sufficient"],
      "3": ["[Foot] Good: Feet are uncrossed correctly"],
      "4": ["[Foot] Good: Crossover is sufficient"]
    },
    "stability": {
      "5": []
    }
  },
  "visualizations": {
    "Step_2_Analysis": "base64_encoded_image_data",
    "Step_3_Analysis": "base64_encoded_image_data",
    "Step_4_Analysis": "base64_encoded_image_data",
    "Step_5_Analysis": "base64_encoded_image_data"
  }
}
```

#### Error Responses

**400 Bad Request**
```json
{
  "detail": "Failed to detect pose in one or more steps."
}
```

**500 Internal Server Error**
```json
{
  "detail": "Video processing error message"
}
```

---

## 🏠 Room Management API

### List Rooms

Retrieves all active rooms, optionally filtered by room type.

```http
GET /rooms?room_type={type}
```

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `room_type` | string | ❌ | Filter by room type: `feedback` or `general` |

#### Response

**Status**: `200 OK`

```json
[
  {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "name": "Bowling Analysis Room",
    "room_type": "feedback",
    "description": "Room for bowling posture discussions",
    "max_users": 10,
    "created_at": "2025-08-15T10:00:00Z",
    "created_by": null,
    "is_active": true
  }
]
```

### Create Room

Creates a new room for users to join and chat.

```http
POST /rooms
```

#### Request

**Content-Type**: `application/json`

```json
{
  "name": "Pleease some feedback on my posture",
  "room_type": "feedback",
  "description": "Discussion room for technique improvement",
  "max_users": 15
}
```

#### Parameters

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `name` | string | ✅ | - | Room display name |
| `room_type` | string | ❌ | `feedback` | Room type: `feedback` or `general` |
| `description` | string | ❌ | `null` | Room description |
| `max_users` | integer | ❌ | `10` | Maximum number of users |

#### Response

**Status**: `200 OK`

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440001",
  "name": "Pleease some feedback on my posture",
  "room_type": "feedback",
  "description": "Discussion room for technique improvement",
  "max_users": 15,
  "created_at": "2025-08-15T10:05:00Z",
  "created_by": null,
  "is_active": true
}
```

### Get Room Details

Retrieves detailed information about a specific room.

```http
GET /rooms/{room_id}
```

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `room_id` | string | ✅ | UUID of the room |

#### Response

**Status**: `200 OK`

```json
{
  "room": {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "name": "Bowling Analysis Room",
    "room_type": "feedback",
    "description": "Room for bowling posture discussions",
    "max_users": 10,
    "created_at": "2025-08-15T10:00:00Z",
    "created_by": null,
    "is_active": true
  },
  "user_count": 3,
  "users": [
    {
      "client_id": "user123",
      "username": "BowlingPro"
    },
    {
      "client_id": "user456", 
      "username": "User_user456"
    }
  ]
}
```

#### Error Responses

**404 Not Found**
```json
{
  "detail": "Room not found"
}
```

### Join Room

Prepares a user to join a room via WebSocket connection.

```http
POST /rooms/{room_id}/join
```

#### Request

**Content-Type**: `application/json`

```json
{
  "username": "MyUsername"
}
```

#### Parameters

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `username` | string | ❌ | Display name for the user |

#### Response

**Status**: `200 OK`

```json
{
  "success": true,
  "room": {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "name": "Bowling Analysis Room",
    "room_type": "feedback",
    "description": "Room for bowling posture discussions",
    "max_users": 10,
    "created_at": "2025-08-15T10:00:00Z",
    "created_by": null,
    "is_active": true
  },
  "user_count": 2,
  "websocket_url": "/ws/550e8400-e29b-41d4-a716-446655440000/{client_id}"
}
```

#### Error Responses

**400 Bad Request**
```json
{
  "detail": "Room is not active"
}
```

**404 Not Found**
```json
{
  "detail": "Room not found"
}
```

---

## 💬 WebSocket Communication

### Room-based Chat Connection

Establishes a WebSocket connection for real-time chat within a specific room.

```websocket
WS /ws/{room_id}/{client_id}?username={username}
```

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `room_id` | string | ✅ | UUID of the room to join |
| `client_id` | string | ✅ | Unique identifier for the client |
| `username` | string | ❌ | Display name (defaults to `User_{client_id[:8]}`) |

#### Connection Flow

1. **Connection**: Client connects to WebSocket endpoint
2. **Room Validation**: Server validates room exists and is active
3. **User Registration**: Client is added to room and WebSocket manager
4. **Welcome Message**: Server sends room info and user list
5. **Presence Broadcast**: Other users notified of new user

#### Message Types

### Incoming Messages (Client → Server)

#### Chat Message
```json
{
  "type": "chat_message",
  "message": "Hello everyone!"
}
```

#### Plain Text (Legacy)
```json
"Hello everyone!"
```

### Outgoing Messages (Server → Client)

#### Chat Message
```json
{
  "type": "message",
  "data": {
    "id": "msg-uuid",
    "room_id": "room-uuid",
    "client_id": "user123",
    "username": "BowlingPro",
    "message": "Hello everyone!",
    "timestamp": "2025-08-15T10:15:00Z"
  }
}
```

#### User Joined
```json
{
  "type": "user_joined",
  "data": {
    "client_id": "user456",
    "username": "NewUser",
    "message": "NewUser님이 Bowling Analysis Room에 입장했습니다"
  }
}
```

#### User Left
```json
{
  "type": "user_left",
  "data": {
    "client_id": "user456",
    "username": "NewUser",
    "message": "NewUser님이 Bowling Analysis Room에서 나갔습니다"
  }
}
```

#### User List Update
```json
{
  "type": "user_list",
  "data": {
    "users": [
      {
        "client_id": "user123",
        "username": "BowlingPro"
      }
    ]
  }
}
```

#### Room Information
```json
{
  "type": "room_info",
  "data": {
    "room": {
      "id": "room-uuid",
      "name": "Bowling Analysis Room",
      "room_type": "feedback",
      "description": "Room for discussions",
      "max_users": 10,
      "created_at": "2025-08-15T10:00:00Z",
      "created_by": null,
      "is_active": true
    },
    "users": [...],
    "user_count": 3
  }
}
```

#### Connection Errors

**Room Not Found (4004)**
```
Connection closed: Room not found or inactive
```

**Room Full (4003)**
```
Connection closed: Cannot join room (may be full)
```

---

## 🔧 Utility Endpoints

### Health Check

Basic health check endpoint to verify server status.

```http
GET /
```

#### Response

**Status**: `200 OK`

```json
{
  "message": "Bowling Posture Guide API is running."
}
```

---

## 📊 Data Models

### Core Analysis Models

#### AnalysisResult
```typescript
interface AnalysisResult {
  feedback: AnalysisFeedback;
  visualizations: Record<string, string>;
}
```

#### AnalysisFeedback
```typescript
interface AnalysisFeedback {
  torso: Record<number, string[]>;    // Steps 2,3,4,5
  foot: Record<number, string[]>;     // Steps 2,3,4
  stability: Record<number, string[]>; // Step 5
}
```

### Room Management Models

#### Room
```typescript
interface Room {
  id: string;
  name: string;
  room_type: "feedback" | "general";
  description?: string;
  max_users?: number;
  created_at: string;
  created_by?: string;
  is_active: boolean;
}
```

#### RoomCreateRequest
```typescript
interface RoomCreateRequest {
  name: string;
  room_type?: "feedback" | "general";
  description?: string;
  max_users?: number;
}
```

#### RoomInfo
```typescript
interface RoomInfo {
  room: Room;
  user_count: number;
  users: Array<{
    client_id: string;
    username: string;
  }>;
}
```

### Chat Models

#### ChatMessage
```typescript
interface ChatMessage {
  id: string;
  room_id: string;
  client_id: string;
  username: string;
  message: string;
  timestamp: string;
  message_type?: string;
}
```

#### ChatResponse
```typescript
interface ChatResponse {
  type: "message" | "user_joined" | "user_left" | "user_list" | "room_info";
  data: Record<string, any>;
}
```

---

## 🔐 Authentication & Security

### Current Implementation
- **No Authentication**: Currently open access for development
- **CORS**: Configured for localhost origins
- **Input Validation**: Pydantic models validate all inputs
- **File Upload**: Basic file type and size validation

### Security Considerations
- **File Upload**: Virus scanning recommended for production
- **Rate Limiting**: Consider implementing for video analysis endpoint
- **Authentication**: JWT or session-based auth recommended for production
- **HTTPS**: Required for production WebSocket connections

---

## 🚦 Rate Limits & Quotas

### Current Limits
- **Video Analysis**: No limits (consider adding for production)
- **Room Creation**: No limits (consider user-based limits)
- **WebSocket Connections**: No limits per user
- **File Upload**: Default FastAPI multipart limits

### Recommended Production Limits
- **Video Analysis**: 10 requests/hour per IP
- **Room Creation**: 5 rooms/hour per user
- **File Size**: 100MB maximum video size
- **WebSocket**: 5 concurrent connections per user

---

## 📈 Monitoring & Logging

### Available Logs
- **Request Logs**: FastAPI access logs
- **Error Logs**: Python exception tracking
- **WebSocket Events**: Connection/disconnection events
- **Room Events**: Room creation, user join/leave

### Health Monitoring
```http
GET /health
```
*Note: Not currently implemented - recommended addition*

### Metrics Endpoints
```http
GET /metrics
```
*Note: Not currently implemented - recommended for production*

---

## 🔄 API Versioning

### Current Version
- **Version**: v1 (implicit)
- **Versioning Strategy**: URL path versioning (not yet implemented)
- **Backwards Compatibility**: Breaking changes will increment version

### Future Versioning
```http
GET /v1/rooms
GET /v2/rooms
```

---

## 📝 Examples & SDKs

### JavaScript/React Example
```javascript
// Video Analysis
const analyzeVideo = async (videoFile, timestamps) => {
  const formData = new FormData();
  formData.append('video', videoFile);
  formData.append('timestamps', JSON.stringify(timestamps));
  
  const response = await fetch('/analyze/interactive-steps', {
    method: 'POST',
    body: formData
  });
  
  return await response.json();
};

// WebSocket Connection
const connectToRoom = (roomId, clientId, username) => {
  const ws = new WebSocket(
    `ws://localhost:8000/ws/${roomId}/${clientId}?username=${username}`
  );
  
  ws.onmessage = (event) => {
    const message = JSON.parse(event.data);
    console.log('Received:', message);
  };
  
  return ws;
};
```

### Python Example
```python
import requests
import json

# Create Room
def create_room(name, room_type="feedback"):
    response = requests.post('http://localhost:8000/rooms', json={
        'name': name,
        'room_type': room_type,
        'max_users': 10
    })
    return response.json()

# Upload Video Analysis
def analyze_video(video_path, timestamps):
    with open(video_path, 'rb') as video_file:
        response = requests.post(
            'http://localhost:8000/analyze/interactive-steps',
            files={'video': video_file},
            data={'timestamps': json.dumps(timestamps)}
        )
    return response.json()
```

---

*Last Updated: 2025-08-15*  
*API Version: 1.0*