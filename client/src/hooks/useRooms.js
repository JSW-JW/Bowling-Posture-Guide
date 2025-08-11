import { useState, useEffect } from 'react';

const API_BASE_URL = 'http://localhost:8000';

export const useRooms = () => {
  const [rooms, setRooms] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchRooms = async (roomType = null) => {
    setLoading(true);
    setError(null);
    try {
      const url = roomType 
        ? `${API_BASE_URL}/rooms?room_type=${roomType}` 
        : `${API_BASE_URL}/rooms`;
      
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error('Failed to fetch rooms');
      }
      
      const data = await response.json();
      setRooms(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const createRoom = async (roomData) => {
    setError(null);
    try {
      const response = await fetch(`${API_BASE_URL}/rooms`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(roomData),
      });

      if (!response.ok) {
        throw new Error('Failed to create room');
      }

      const newRoom = await response.json();
      setRooms(prev => [...prev, newRoom]);
      return newRoom;
    } catch (err) {
      setError(err.message);
      throw err;
    }
  };

  const joinRoom = async (roomId) => {
    setError(null);
    try {
      const response = await fetch(`${API_BASE_URL}/rooms/${roomId}/join`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({}),
      });

      if (!response.ok) {
        throw new Error('Failed to join room');
      }

      const result = await response.json();
      return result;
    } catch (err) {
      setError(err.message);
      throw err;
    }
  };

  const getRoomInfo = async (roomId) => {
    setError(null);
    try {
      const response = await fetch(`${API_BASE_URL}/rooms/${roomId}`);
      if (!response.ok) {
        throw new Error('Failed to get room info');
      }

      const roomInfo = await response.json();
      return roomInfo;
    } catch (err) {
      setError(err.message);
      throw err;
    }
  };

  // Auto-fetch rooms on mount
  useEffect(() => {
    fetchRooms();
  }, []);

  return {
    rooms,
    loading,
    error,
    fetchRooms,
    createRoom,
    joinRoom,
    getRoomInfo,
  };
};