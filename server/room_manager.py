from typing import Dict, List, Optional
from datetime import datetime
import uuid
from models import Room, RoomType, RoomCreateRequest

class RoomManager:
    """
    In-memory room management system.
    Future: Replace with database repository pattern for persistence.
    """
    
    def __init__(self):
        self.rooms: Dict[str, Room] = {}
        self.user_rooms: Dict[str, str] = {}  # client_id -> room_id mapping
        
    def create_room(self, request: RoomCreateRequest, created_by: Optional[str] = None) -> Room:
        """Create a new room. Future: Save to database."""
        room_id = str(uuid.uuid4())
        room = Room(
            id=room_id,
            name=request.name,
            room_type=request.room_type,
            description=request.description,
            max_users=request.max_users,
            created_at=datetime.now(),
            created_by=created_by,
            is_active=True
        )
        
        self.rooms[room_id] = room
        return room
    
    def get_room(self, room_id: str) -> Optional[Room]:
        """Get room by ID. Future: Query from database."""
        return self.rooms.get(room_id)
    
    def list_rooms(self, room_type: Optional[RoomType] = None, is_active: bool = True) -> List[Room]:
        """List all rooms. Future: Database query with pagination."""
        rooms = []
        for room in self.rooms.values():
            if room.is_active == is_active:
                if room_type is None or room.room_type == room_type:
                    rooms.append(room)
        return rooms
    
    def join_room(self, client_id: str, room_id: str) -> bool:
        """Join a user to a room. Future: Check permissions, max users, etc."""
        room = self.get_room(room_id)
        if not room or not room.is_active:
            return False
            
        # Check max users limit
        if room.max_users:
            current_users = self.get_room_users(room_id)
            if len(current_users) >= room.max_users:
                return False
        
        # Remove user from previous room if any
        self.leave_room(client_id)
        
        # Join new room
        self.user_rooms[client_id] = room_id
        return True
    
    def leave_room(self, client_id: str) -> Optional[str]:
        """Leave current room. Returns the room_id that was left."""
        if client_id in self.user_rooms:
            room_id = self.user_rooms[client_id]
            del self.user_rooms[client_id]
            return room_id
        return None
    
    def get_user_room(self, client_id: str) -> Optional[str]:
        """Get the room ID that the user is currently in."""
        return self.user_rooms.get(client_id)
    
    def get_room_users(self, room_id: str) -> List[str]:
        """Get all users in a specific room."""
        return [client_id for client_id, r_id in self.user_rooms.items() if r_id == room_id]
    
    def deactivate_room(self, room_id: str) -> bool:
        """Deactivate a room. Future: Update database record."""
        if room_id in self.rooms:
            self.rooms[room_id].is_active = False
            
            # Remove all users from the room
            users_to_remove = [client_id for client_id, r_id in self.user_rooms.items() if r_id == room_id]
            for client_id in users_to_remove:
                del self.user_rooms[client_id]
            
            return True
        return False
    
    def get_room_stats(self, room_id: str) -> Dict:
        """Get room statistics. Future: Include database metrics."""
        room = self.get_room(room_id)
        if not room:
            return {}
        
        users = self.get_room_users(room_id)
        return {
            "room_id": room_id,
            "name": room.name,
            "type": room.room_type,
            "user_count": len(users),
            "max_users": room.max_users,
            "is_active": room.is_active,
            "created_at": room.created_at.isoformat()
        }

# Global room manager instance
# Future: Inject as dependency with database repository
room_manager = RoomManager()