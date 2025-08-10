from typing import List, Dict
from fastapi import WebSocket
import redis.asyncio as redis
import asyncio
import json

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections.values():
            await connection.send_text(message)

class RedisPubSubManager:
    def __init__(self, manager: ConnectionManager):
        self.redis_client = None
        self.pubsub = None
        self.manager = manager
        self.channel = "chat_channel"

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
        if not self.pubsub:
            return
        await self.pubsub.subscribe(self.channel)
        print(f"Subscribed to Redis channel: {self.channel}")

    async def publish(self, message: str):
        if not self.redis_client:
            # Redis 연결이 없는 경우, 로컬 브로드캐스트만 수행 (Graceful fallback)
            await self.manager.broadcast(message)
            return
        await self.redis_client.publish(self.channel, message)

    async def listen(self):
        if not self.pubsub:
            return
        print("Listening for messages on Redis channel...")
        while True:
            try:
                message = await self.pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                if message:
                    print(f"Received message from Redis: {message['data']}")
                    await self.manager.broadcast(message['data'])
            except asyncio.CancelledError:
                print("Listener task cancelled.")
                break
            except Exception as e:
                print(f"Error in Redis listener: {e}")
                await asyncio.sleep(1) # 재연결 시도 전 잠시 대기

    async def close(self):
        if self.pubsub:
            await self.pubsub.close()
        if self.redis_client:
            await self.redis_client.close()
        print("Redis connection closed.")

# 전역 인스턴스 생성
manager = ConnectionManager()
pubsub_manager = RedisPubSubManager(manager)
