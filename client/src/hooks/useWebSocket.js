import { useState, useEffect, useRef, useCallback } from 'react';

const useWebSocket = (clientId, username, roomId) => {
  const [messages, setMessages] = useState([]);
  const [users, setUsers] = useState([]);
  const [roomInfo, setRoomInfo] = useState(null);
  const [connectionStatus, setConnectionStatus] = useState('Disconnected');
  const ws = useRef(null);
  const isMounted = useRef(true); // ✅ cleanup을 위한 마운트 상태 추적
  const retryCount = useRef(0);
  const maxRetries = useRef(5);
  const retryTimeout = useRef(null);

  const connect = useCallback(() => {
    // Prevent duplicate connections
    if (ws.current?.readyState === WebSocket.OPEN || !roomId || !isMounted.current) {
      console.log("log 1")
      return;
    }

    // Close any existing connection first
    if (ws.current) {
      ws.current.close();
    }

    // Clear any pending reconnection attempts
    if (retryTimeout.current) {
      clearTimeout(retryTimeout.current);
      retryTimeout.current = null;
    }

    try {
      console.log(`Connecting to room: ${roomId} as ${username}`);
      const wsBaseUrl = process.env.REACT_APP_WS_BASE_URL || 'ws://localhost:8000';
      const wsUrl = `${wsBaseUrl}/ws/${roomId}/${clientId}?username=${encodeURIComponent(username)}`;
      ws.current = new WebSocket(wsUrl);

      ws.current.onopen = () => {
        console.log('WebSocket connected');
        if (isMounted.current) { // ✅ 마운트 상태 체크
          setConnectionStatus('Connected');
        }
        retryCount.current = 0; // Reset retry count on successful connection
      };

      ws.current.onmessage = (event) => {
        if (!isMounted.current) return; // ✅ 언마운트된 경우 조기 리턴
        
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
        if (isMounted.current) { // ✅ 마운트 상태 체크
          setConnectionStatus('Disconnected');
          
          // Only retry if not manually closed and under retry limit and component is mounted
          if (event.code !== 1000 && retryCount.current < maxRetries.current && roomId) {
            retryCount.current++;
            const delay = Math.min(1000 * Math.pow(2, retryCount.current), 30000); // Exponential backoff with max 30s
            console.log(`Retrying connection in ${delay}ms (attempt ${retryCount.current}/${maxRetries.current})`);
            
            retryTimeout.current = setTimeout(() => {
              if (isMounted.current) { // 재연결 시에도 마운트 상태 재확인
                connect();
              }
            }, delay);
          } else if (retryCount.current >= maxRetries.current) {
            console.log('Max retry attempts reached');
            setConnectionStatus('Failed');
          }
        }
      };

      ws.current.onerror = (error) => {
        console.error('WebSocket error:', error);
        if (isMounted.current) { // ✅ 마운트 상태 체크
          setConnectionStatus('Error');
        }
      };

    } catch (error) {
      console.error('Error creating WebSocket connection:', error);
      if (isMounted.current) { // ✅ 마운트 상태 체크
        setConnectionStatus('Error');
      }
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
    // ✅ 재연결 타이머 정리 (both retry and reconnect timeouts)
    if (retryTimeout.current) {
      clearTimeout(retryTimeout.current);
      retryTimeout.current = null;
    }
    
    // ✅ WebSocket 연결 정리
    if (ws.current) {
      ws.current.close(1000, 'User disconnected'); // Normal closure
      ws.current = null;
    }
    retryCount.current = 0;
  }, []);

  // Clean up messages when room changes
  useEffect(() => {
    if (isMounted.current) { // ✅ 마운트 상태 체크
      setMessages([]);
      setUsers([]);
      setRoomInfo(null);
    }
  }, [roomId]);

  // Connection effect - handles room changes
  useEffect(() => {
    // ✅ effect 시작 시 마운트 상태 재설정
    isMounted.current = true;
    
    console.log("🔍 useEffect triggered - roomId:", roomId, "type:", typeof roomId);
    if (roomId) {
      console.log("✅ Connecting to room:", roomId);
      connect();
    } else {
      console.log("❌ No roomId, disconnecting...");
      disconnect();
      if (isMounted.current) { // ✅ 마운트 상태 체크
        setConnectionStatus('Disconnected');
      }
    }
  }, [connect, disconnect, roomId]);

  // Cleanup effect - only runs on component unmount
  useEffect(() => {
    return () => {
      isMounted.current = false;
      disconnect();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

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