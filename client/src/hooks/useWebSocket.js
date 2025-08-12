import { useState, useEffect, useRef, useCallback } from 'react';

const useWebSocket = (clientId, username, roomId) => {
  const [messages, setMessages] = useState([]);
  const [users, setUsers] = useState([]);
  const [roomInfo, setRoomInfo] = useState(null);
  const [connectionStatus, setConnectionStatus] = useState('Disconnected');
  const ws = useRef(null);
  const retryCount = useRef(0);
  const maxRetries = useRef(5);
  const retryTimeout = useRef(null);

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
      const wsBaseUrl = process.env.REACT_APP_WS_BASE_URL || 'ws://localhost:8000';
      const wsUrl = `${wsBaseUrl}/ws/${roomId}/${clientId}?username=${encodeURIComponent(username)}`;
      ws.current = new WebSocket(wsUrl);

      ws.current.onopen = () => {
        console.log('WebSocket connected');
        setConnectionStatus('Connected');
        retryCount.current = 0; // Reset retry count on successful connection
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

      ws.current.onclose = (event) => {
        console.log('WebSocket disconnected', event.code, event.reason);
        setConnectionStatus('Disconnected');
        
        // Only retry if not manually closed and under retry limit
        if (event.code !== 1000 && retryCount.current < maxRetries.current) {
          retryCount.current++;
          const delay = Math.min(1000 * Math.pow(2, retryCount.current), 30000); // Exponential backoff with max 30s
          console.log(`Retrying connection in ${delay}ms (attempt ${retryCount.current}/${maxRetries.current})`);
          
          retryTimeout.current = setTimeout(() => {
            connect();
          }, delay);
        } else if (retryCount.current >= maxRetries.current) {
          console.log('Max retry attempts reached');
          setConnectionStatus('Failed');
        }
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
    if (retryTimeout.current) {
      clearTimeout(retryTimeout.current);
      retryTimeout.current = null;
    }
    if (ws.current) {
      ws.current.close(1000, 'User disconnected'); // Normal closure
    }
    retryCount.current = 0;
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