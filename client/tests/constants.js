// tests/constants.js
// UI Text Constants
export const UI_TEXT = {
  // Main page
  MAIN_TITLE: '🎳 볼링 자세 분석 AI',
  MAIN_SUBTITLE: '동영상을 재생하며 5번의 스텝이 끝나는 지점에서 \'s\' 키를 누르세요.',
  
  // Keyboard shortcuts
  SHORTCUTS: {
    PLAY_PAUSE: 'Space : 재생/일시정지',
    BACKWARD: '← : 뒤로 1프레임',
    FORWARD: '→ : 앞으로 1프레임',
    MARK_STEP: 's : 스텝 지정'
  },
  
  // Room management
  ROOM: {
    JOIN_BUTTON: '🚪 Feedback Room 입장',
    JOINED_PREFIX: '📍 ',
    MODAL_TITLE: '🎳 Feedback Room 선택',
    EXISTING_ROOMS: '활성 Feedback Room',
    CREATE_NEW: '새 Feedback Room 만들기',
    NAME_PLACEHOLDER: '예: 볼링 자세 피드백 방',
    CREATE_BUTTON: '방 만들기',
    CREATING_BUTTON: '생성 중...',
    CREATE_ROOM_BTN: '➕ 방 만들기',
    REFRESH_BTN: '🔄 새로고침'
  },
  
  // Chat system
  CHAT: {
    NO_ROOM_MESSAGE: 'Feedback Room에 입장해야 채팅을 사용할 수 있습니다.',
    MESSAGE_PLACEHOLDER: '메시지를 입력하세요...',
    JOIN_MESSAGE_SUFFIX: '님이 입장하셨습니다',
    SEND_BUTTON: '전송'
  },
  
  // Analysis
  ANALYSIS: {
    BUTTON_ANALYZE: '분석하기',
    BUTTON_ANALYZING: '분석 중...',
    RESULTS_TITLE: '분석 결과',
    FEEDBACK_TITLE: '자세 피드백',
    VISUALIZATION_TITLE: '분석 시각화'
  }
};

// Selectors
export const SELECTORS = {
  // Main elements
  MAIN_TITLE: 'h1',
  FILE_INPUT: 'input[type="file"]',
  VIDEO_ELEMENT: 'video',
  
  // User interface
  USER_INFO: '.user-info',
  USERNAME_DISPLAY: '.username-display',
  ROOM_BTN: '.room-btn',
  CONTROLS_GUIDE: '.controls-guide',
  KEY_CHIP: '.key-chip',
  
  // Room management
  ROOM_SELECTION_OVERLAY: '.room-selection-overlay',
  ROOM_SELECTION: '.room-selection-modal',
  ROOM_SELECTION_TITLE: '.room-selection-header h2',
  EXISTING_ROOMS: '.rooms-list',
  EXISTING_ROOMS_TITLE: '.rooms-list h3',
  ROOMS_LIST: '.rooms-list',
  CREATE_ROOM: '.create-room-form',
  CREATE_ROOM_TITLE: '.create-room-form h3',
  ROOM_NAME_INPUT: '.create-room-form input[type="text"]',
  CREATE_SUBMIT_BTN: '.create-room-form button[type="submit"]',
  CREATE_ROOM_BTN: '.create-room-btn',
  CLOSE_BTN: '.close-btn',
  JOIN_BTN: '.join-btn',
  ROOM_ITEM: '.room-item',
  
  // Chat system
  CHAT_CONTAINER: '.chat-container',
  CHAT_TOGGLE: '.toggle-btn',
  NO_ROOM_STATE: '.chat-no-room',
  ROOM_INFO: '.chat-title',
  MESSAGES: '.messages-list',
  MESSAGE: '.message',
  SYSTEM_MESSAGE: '.system-message',
  MESSAGE_INPUT: '.message-input',
  SEND_BTN: '.send-button',
  USERS_LIST: '.users-list',
  USER: '.user-badge',
  USER_COUNT: '.user-count',
  LEAVE_ROOM_BTN: '.leave-room-btn',
  
  // Analysis
  VIDEO_CONTAINER: '.video-container',
  STATUS: '.status',
  TIMESTAMP_CHIP: '.timestamp-chip',
  ANALYZE_BTN: 'button:has-text("분석하기")',
  ERROR_MESSAGE: '.error-message',
  RESULTS_SECTION: '.results-section',
  FEEDBACK_CONTAINER: '.feedback-container',
  VISUALS_CONTAINER: '.visuals-container',
  VISUALS_GRID: '.visuals-grid',
  VISUAL_ITEM: '.visual-item'
};

// Test data
export const TEST_DATA = {
  ROOMS: {
    TEST_ROOM: '테스트 룸',
    CHAT_TEST_ROOM: '채팅 테스트 룸',
    BOWLING_FEEDBACK_ROOM: '볼링 피드백 룸',
    MESSAGE_TEST_ROOM: '메시지 테스트',
    INPUT_TEST_ROOM: '입력 테스트',
    WEBSOCKET_TEST_ROOM: 'WebSocket 테스트'
  },
  
  MESSAGES: {
    TEST_MESSAGE: '안녕하세요! 테스트 메시지입니다.',
    ENTER_KEY_MESSAGE: 'Enter키로 전송하는 메시지'
  },
  
  DEFAULT_MAX_USERS: 5,
  DEFAULT_MAX_USERS_STRING: '5'
};

// Regular expressions for dynamic content
export const PATTERNS = {
  USERNAME: /👤 볼러\d+_\w+/,
  JOIN_MESSAGE: /입장했습니다|joined/,
  STEP_TIMESTAMP: /스텝 \d+: \d+\.\d+초/
};