import React, { useState, useRef, useEffect } from 'react';
import useWebSocket from '../hooks/useWebSocket';
import './Chat.css';

const Chat = ({ clientId, username, roomId, roomInfo, isVisible, onToggle, onLeaveRoom }) => {
  const [inputMessage, setInputMessage] = useState('');
  const messagesEndRef = useRef(null);
  const { messages, users, connectionStatus, sendMessage } = useWebSocket(clientId, username, roomId);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (inputMessage.trim()) {
      const success = sendMessage(inputMessage.trim());
      if (success) {
        setInputMessage('');
      } else {
        alert('Failed to send message. Please check your connection.');
      }
    }
  };

  const formatTime = (timestamp) => {
    return new Date(timestamp).toLocaleTimeString('ko-KR', {
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const getConnectionStatusColor = () => {
    switch (connectionStatus) {
      case 'Connected': return '#4CAF50';
      case 'Connecting': return '#FF9800';
      case 'Disconnected': return '#f44336';
      case 'Error': return '#f44336';
      default: return '#999';
    }
  };

  if (!roomId) {
    return (
      <div className={`chat-container ${isVisible ? 'visible' : 'hidden'}`}>
        <div className="chat-header">
          <div className="chat-title">
            <span>💬 채팅</span>
          </div>
          <div className="chat-controls">
            <button className="toggle-btn" onClick={onToggle}>
              {isVisible ? '▼' : '▲'}
            </button>
          </div>
        </div>
        {isVisible && (
          <div className="chat-no-room">
            <p>Feedback Room에 입장해야 채팅을 사용할 수 있습니다.</p>
          </div>
        )}
      </div>
    );
  }

  return (
    <div className={`chat-container ${isVisible ? 'visible' : 'hidden'}`}>
      <div className="chat-header">
        <div className="chat-title">
          <span>💬 {roomInfo?.name || '채팅'}</span>
          <span 
            className="connection-status"
            style={{ color: getConnectionStatusColor() }}
          >
            {connectionStatus}
          </span>
        </div>
        <div className="chat-controls">
          <span className="user-count">👥 {users.length}</span>
          {onLeaveRoom && (
            <button className="leave-room-btn" onClick={onLeaveRoom} title="방 나가기">
              🚪
            </button>
          )}
          <button className="toggle-btn" onClick={onToggle}>
            {isVisible ? '▼' : '▲'}
          </button>
        </div>
      </div>

      {isVisible && (
        <>
          <div className="users-list">
            <h4>접속자 ({users.length})</h4>
            <div className="users-grid">
              {users.map(user => (
                <span key={user.client_id} className="user-badge">
                  {user.username}
                </span>
              ))}
            </div>
          </div>

          <div className="messages-container">
            <div className="messages-list">
              {messages.map((message) => (
                <div
                  key={message.id}
                  className={`message ${message.isSystem ? 'system-message' : ''} ${
                    message.client_id === clientId ? 'own-message' : ''
                  }`}
                >
                  {message.isSystem ? (
                    <div className="system-content">
                      <span className="system-text">{message.message}</span>
                      <span className="message-time">{formatTime(message.timestamp)}</span>
                    </div>
                  ) : (
                    <div className="message-content">
                      <div className="message-header">
                        <span className="sender-name">{message.username}</span>
                        <span className="message-time">{formatTime(message.timestamp)}</span>
                      </div>
                      <div className="message-text">{message.message}</div>
                    </div>
                  )}
                </div>
              ))}
              <div ref={messagesEndRef} />
            </div>
          </div>

          <form onSubmit={handleSubmit} className="message-input-form">
            <div className="input-container">
              <input
                type="text"
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                placeholder="메시지를 입력하세요..."
                className="message-input"
                disabled={connectionStatus !== 'Connected'}
              />
              <button
                type="submit"
                className="send-button"
                disabled={!inputMessage.trim() || connectionStatus !== 'Connected'}
              >
                전송
              </button>
            </div>
          </form>
        </>
      )}
    </div>
  );
};

export default Chat;