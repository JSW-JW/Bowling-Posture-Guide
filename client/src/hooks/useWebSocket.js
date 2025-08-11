import { useState, useEffect, useRef, useCallback } from 'react';

const useWebSocket = (clientId, username, roomId) => {
  const [messages, setMessages] = useState([]);
  const [users, setUsers] = useState([]);
  const [roomInfo, setRoomInfo] = useState(null);
  const [connectionStatus, setConnectionStatus] = useState('Disconnected');
  const ws = useRef(null);

  const connect = useCallback(() => {
    // Prevent duplicate connections
    if (ws.current?.readyState === WebSocket.OPEN || !roomId) {
      return;
    }

    // Close any existing connection first
    if (ws.current) {
      ws.current.close();
    }

    try {
      console.log(`Connecting to room: ${roomId} as ${username}`);
      const wsUrl = `ws://localhost:8000/ws/${roomId}/${clientId}?username=${encodeURIComponent(username)}`;
      ws.current = new WebSocket(wsUrl);

      ws.current.onopen = () => {
        console.log('WebSocket connected');
        setConnectionStatus('Connected');
      };

      ws.current.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          
          switch (data.type) {
            case 'message':
              setMessages(prev => [...prev, data.data]);
              break;
            case 'user_joined':
              setMessages(prev => [...prev, {
                id: `system-${Date.now()}`,
                message: data.data.message,
                timestamp: new Date().toISOString(),
                isSystem: true
              }]);
              break;
            case 'user_left':
              setMessages(prev => [...prev, {
                id: `system-${Date.now()}`,
                message: data.data.message,
                timestamp: new Date().toISOString(),
                isSystem: true
              }]);
              break;
            case 'user_list':
              setUsers(data.data.users);
              break;
            case 'room_info':
              setRoomInfo(data.data.room);
              setUsers(data.data.users);
              break;
            default:
              console.log('Unknown message type:', data.type);
          }
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };

      ws.current.onclose = () => {
        console.log('WebSocket disconnected');
        setConnectionStatus('Disconnected');
        // Retry connection after 3 seconds
        setTimeout(connect, 3000);
      };

      ws.current.onerror = (error) => {
        console.error('WebSocket error:', error);
        setConnectionStatus('Error');
      };

    } catch (error) {
      console.error('Error creating WebSocket connection:', error);
      setConnectionStatus('Error');
    }
  }, [clientId, username, roomId]);

  const sendMessage = useCallback((message) => {
    if (ws.current?.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify({
        type: 'chat_message',
        message: message
      }));
      return true;
    }
    return false;
  }, []);

  const disconnect = useCallback(() => {
    if (ws.current) {
      ws.current.close();
    }
  }, []);

  // Clean up messages when room changes
  useEffect(() => {
    setMessages([]);
    setUsers([]);
    setRoomInfo(null);
  }, [roomId]);

  useEffect(() => {
    if (roomId) {
      connect();
    } else {
      disconnect();
      setConnectionStatus('Disconnected');
    }
    
    return () => {
      disconnect();
    };
  }, [connect, disconnect, roomId]);

  return {
    messages,
    users,
    roomInfo,
    connectionStatus,
    sendMessage,
    connect,
    disconnect
  };
};

export default useWebSocket;