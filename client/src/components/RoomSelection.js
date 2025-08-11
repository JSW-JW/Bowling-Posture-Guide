import React, { useState } from 'react';
import { useRooms } from '../hooks/useRooms';
import './RoomSelection.css';

const RoomSelection = ({ onRoomSelect, onClose }) => {
  const { rooms, loading, error, createRoom, fetchRooms } = useRooms();
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [newRoomName, setNewRoomName] = useState('');
  const [newRoomDescription, setNewRoomDescription] = useState('');
  const [isCreating, setIsCreating] = useState(false);

  const handleJoinRoom = async (room) => {
    try {
      onRoomSelect(room);
    } catch (error) {
      alert('방 참가에 실패했습니다: ' + error.message);
    }
  };

  const handleCreateRoom = async (e) => {
    e.preventDefault();
    if (!newRoomName.trim()) {
      alert('방 이름을 입력해주세요.');
      return;
    }

    setIsCreating(true);
    try {
      const roomData = {
        name: newRoomName.trim(),
        room_type: 'feedback',
        description: newRoomDescription.trim() || null,
        max_users: 10
      };

      const newRoom = await createRoom(roomData);
      setNewRoomName('');
      setNewRoomDescription('');
      setShowCreateForm(false);
      onRoomSelect(newRoom);
    } catch (error) {
      alert('방 생성에 실패했습니다: ' + error.message);
    } finally {
      setIsCreating(false);
    }
  };

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleString('ko-KR');
  };

  return (
    <div className="room-selection-overlay">
      <div className="room-selection-modal">
        <div className="room-selection-header">
          <h2>🎳 Feedback Room 선택</h2>
          <button className="close-btn" onClick={onClose}>×</button>
        </div>

        <div className="room-selection-content">
          {error && <div className="error-message">{error}</div>}

          <div className="room-actions">
            <button 
              className="refresh-btn"
              onClick={() => fetchRooms()}
              disabled={loading}
            >
              {loading ? '새로고침 중...' : '🔄 새로고침'}
            </button>
            <button 
              className="create-room-btn"
              onClick={() => setShowCreateForm(!showCreateForm)}
            >
              {showCreateForm ? '취소' : '➕ 방 만들기'}
            </button>
          </div>

          {showCreateForm && (
            <form className="create-room-form" onSubmit={handleCreateRoom}>
              <h3>새 Feedback Room 만들기</h3>
              <div className="form-group">
                <label>방 이름 *</label>
                <input
                  type="text"
                  value={newRoomName}
                  onChange={(e) => setNewRoomName(e.target.value)}
                  placeholder="예: 볼링 자세 피드백 방"
                  maxLength={50}
                  required
                />
              </div>
              <div className="form-group">
                <label>방 설명</label>
                <textarea
                  value={newRoomDescription}
                  onChange={(e) => setNewRoomDescription(e.target.value)}
                  placeholder="방에 대한 간단한 설명을 입력하세요"
                  maxLength={200}
                  rows={3}
                />
              </div>
              <div className="form-actions">
                <button type="submit" disabled={isCreating}>
                  {isCreating ? '생성 중...' : '방 만들기'}
                </button>
              </div>
            </form>
          )}

          <div className="rooms-list">
            <h3>활성 Feedback Room ({rooms.length})</h3>
            
            {loading && <div className="loading">방 목록을 불러오는 중...</div>}
            
            {!loading && rooms.length === 0 && (
              <div className="empty-state">
                <p>활성화된 방이 없습니다.</p>
                <p>새로운 Feedback Room을 만들어보세요!</p>
              </div>
            )}

            {rooms.map(room => (
              <div key={room.id} className="room-item">
                <div className="room-info">
                  <div className="room-header">
                    <h4>{room.name}</h4>
                    <span className="room-type">{room.room_type}</span>
                  </div>
                  {room.description && (
                    <p className="room-description">{room.description}</p>
                  )}
                  <div className="room-meta">
                    <span className="room-created">
                      생성일: {formatDate(room.created_at)}
                    </span>
                    <span className="room-capacity">
                      최대 인원: {room.max_users || '무제한'}명
                    </span>
                  </div>
                </div>
                <div className="room-actions">
                  <button 
                    className="join-btn"
                    onClick={() => handleJoinRoom(room)}
                  >
                    입장하기
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default RoomSelection;