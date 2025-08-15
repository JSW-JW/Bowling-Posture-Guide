from typing import List, Dict
from fastapi import WebSocket
import redis.asyncio as redis
import asyncio
import json
import uuid
from datetime import datetime

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, Dict] = {}  # {client_id: {websocket: ws, username: name, room_id: room}}

    async def connect(self, websocket: WebSocket, client_id: str, username: str = None, room_id: str = None):
        await websocket.accept()
        self.active_connections[client_id] = {
            "websocket": websocket,
            "username": username or f"User_{client_id[:8]}",
            "room_id": room_id
        }

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]

    def update_user_room(self, client_id: str, room_id: str):
        """Update user's room assignment"""
        if client_id in self.active_connections:
            self.active_connections[client_id]["room_id"] = room_id

    async def send_personal_message(self, message: dict, client_id: str):
        if client_id in self.active_connections:
            websocket = self.active_connections[client_id]["websocket"]
            await websocket.send_text(json.dumps(message))

    async def broadcast_to_room(self, message: dict, room_id: str):
        """Broadcast message to all users in a specific room"""
        disconnected_clients = []
        for client_id, connection_info in self.active_connections.items():
            if connection_info.get("room_id") == room_id:
                try:
                    websocket = connection_info["websocket"]
                    await websocket.send_text(json.dumps(message))
                except Exception as e:
                    print(f"Error sending message to client {client_id} in room {room_id}: {e}")
                    disconnected_clients.append(client_id)
        
        # Remove disconnected clients
        for client_id in disconnected_clients:
            self.disconnect(client_id)

    async def broadcast(self, message: dict):
        """Broadcast to all connected users (deprecated for room-based chat)"""
        disconnected_clients = []
        for client_id, connection_info in self.active_connections.items():
            try:
                websocket = connection_info["websocket"]
                await websocket.send_text(json.dumps(message))
            except Exception as e:
                print(f"Error sending message to client {client_id}: {e}")
                disconnected_clients.append(client_id)
        
        # Remove disconnected clients
        for client_id in disconnected_clients:
            self.disconnect(client_id)

    def get_room_users(self, room_id: str):
        """Get all users in a specific room"""
        return [
            {"client_id": client_id, "username": info["username"]}
            for client_id, info in self.active_connections.items()
            if info.get("room_id") == room_id
        ]

    def get_active_users(self):
        """Get all active users (for backward compatibility)"""
        return [
            {"client_id": client_id, "username": info["username"], "room_id": info.get("room_id")}
            for client_id, info in self.active_connections.items()
        ]

    def get_username(self, client_id: str):
        return self.active_connections.get(client_id, {}).get("username", f"User_{client_id[:8]}")
    
    def get_user_room(self, client_id: str):
        return self.active_connections.get(client_id, {}).get("room_id")

class RedisPubSubManager:
    def __init__(self, manager: ConnectionManager):
        self.redis_client = None
        self.pubsub = None
        self.manager = manager
        self.base_channel = "bowling_chat"  # Base channel for all rooms
        self.subscribed_channels = set()  # Track subscribed room channels

    async def connect_to_redis(self):
        try:
            self.redis_client = redis.from_url("redis://localhost", encoding="utf-8", decode_responses=True)
            await self.redis_client.ping()
            print("Successfully connected to Redis.")
            self.pubsub = self.redis_client.pubsub()
        except Exception as e:
            print(f"Error connecting to Redis: {e}")
            self.redis_client = None

    async def subscribe(self):
        """Subscribe to the general pattern for all room channels"""
        if not self.pubsub:
            return
        # Subscribe to pattern for all room channels
        await self.pubsub.psubscribe(f"{self.base_channel}:*")
        print(f"Subscribed to Redis pattern: {self.base_channel}:*")

    async def subscribe_to_room(self, room_id: str):
        """Subscribe to a specific room channel - Not needed due to pattern subscription"""
        # Pattern subscription already covers all room channels
        # This method is kept for compatibility but does nothing to prevent duplicate messages
        pass

    async def publish_to_room(self, message: dict, room_id: str):
        """Publish message to a specific room channel"""
        if not self.redis_client:
            # Graceful fallback: local broadcast only when Redis unavailable
            await self.manager.broadcast_to_room(message, room_id)
            return
        
        room_channel = f"{self.base_channel}:{room_id}"
        await self.redis_client.publish(room_channel, json.dumps(message))

    async def publish(self, message: dict):
        """Deprecated: Use publish_to_room instead"""
        if not self.redis_client:
            await self.manager.broadcast(message)
            return
        
        # For backward compatibility, publish to general channel
        await self.redis_client.publish(f"{self.base_channel}:general", json.dumps(message))

    async def listen(self):
        if not self.pubsub:
            return
        print("Listening for messages on Redis channels...")
        while True:
            try:
                message = await self.pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                if message:
                    channel = message['channel']
                    print(f"Received message from Redis channel {channel}: {message['data']}")
                    
                    # Extract room_id from channel name
                    room_id = None
                    if channel.startswith(f"{self.base_channel}:"):
                        room_id = channel[len(f"{self.base_channel}:"):]
                    
                    try:
                        message_data = json.loads(message['data'])
                        if room_id:
                            # Broadcast to specific room
                            await self.manager.broadcast_to_room(message_data, room_id)
                        else:
                            # Fallback to global broadcast
                            await self.manager.broadcast(message_data)
                    except json.JSONDecodeError:
                        # Fallback for plain text messages
                        fallback_message = {"type": "message", "data": {"content": message['data']}}
                        if room_id:
                            await self.manager.broadcast_to_room(fallback_message, room_id)
                        else:
                            await self.manager.broadcast(fallback_message)
            except asyncio.CancelledError:
                print("Listener task cancelled.")
                break
            except Exception as e:
                print(f"Error in Redis listener: {e}")
                await asyncio.sleep(1)  # Wait before retry

    async def close(self):
        if self.pubsub:
            await self.pubsub.close()
        if self.redis_client:
            await self.redis_client.close()
        print("Redis connection closed.")

# 전역 인스턴스 생성
manager = ConnectionManager()
pubsub_manager = RedisPubSubManager(manager)
