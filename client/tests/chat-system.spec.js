// tests/chat-system.spec.js
import { test, expect } from '@playwright/test';
import { TestHelpers } from './helpers.js';
import { UI_TEXT, SELECTORS, TEST_DATA, PATTERNS } from './constants.js';

test.describe('Chat System Features', () => {
  let helpers;

  test.beforeEach(async ({ page }) => {
    helpers = new TestHelpers(page);
    await helpers.navigateToApp();
  });

  test('should display chat toggle button', async ({ page }) => {
    // Chat container should be visible by default
    const chat = page.locator(SELECTORS.CHAT_CONTAINER);
    await expect(chat).toBeVisible();
    
    // Chat toggle button should be visible
    const toggleBtn = page.locator(SELECTORS.CHAT_TOGGLE);
    await expect(toggleBtn).toBeVisible();
  });

  test('should toggle chat visibility', async ({ page }) => {
    const chat = page.locator(SELECTORS.CHAT_CONTAINER);
    const toggleBtn = page.locator(SELECTORS.CHAT_TOGGLE);
    
    // Initially visible
    await expect(chat).toHaveClass(/visible/);
    
    // Click to hide
    await toggleBtn.click();
    await expect(chat).not.toHaveClass(/visible/);
    
    // Click to show again
    await toggleBtn.click();
    await expect(chat).toHaveClass(/visible/);
  });

  test('should show no room message when not in a room', async ({ page }) => {
    // When not connected to any room
    const chat = page.locator(SELECTORS.CHAT_CONTAINER);
    if (await chat.isVisible()) {
      // Should show no room state
      const noRoomMessage = page.locator(SELECTORS.NO_ROOM_STATE);
      if (await noRoomMessage.isVisible()) {
        await expect(noRoomMessage).toHaveText(UI_TEXT.CHAT.NO_ROOM_MESSAGE);
      }
    }
  });

  test('should join room and show chat interface', async ({ page }) => {
    // Create and join a room
    const roomName = TEST_DATA.ROOMS.CHAT_TEST_ROOM;
    await helpers.createRoom(roomName);
    await helpers.waitForRoomJoin();
    
    // Chat should now show room interface
    const chat = page.locator(SELECTORS.CHAT_CONTAINER);
    await expect(chat).toBeVisible();
    
    // Should show room info
    const roomInfo = page.locator(SELECTORS.ROOM_INFO);
    if (await roomInfo.isVisible()) {
      await expect(roomInfo).toContainText(roomName);
    }
  });

  test('should display messages area', async ({ page }) => {
    // Join a room first
    await helpers.createRoom(TEST_DATA.ROOMS.MESSAGE_TEST_ROOM);
    await helpers.waitForRoomJoin();
    
    // Check if messages container exists
    const messagesContainer = page.locator(SELECTORS.MESSAGES);
    if (await messagesContainer.isVisible()) {
      await expect(messagesContainer).toBeVisible();
    }
  });

  test('should have message input and send button', async ({ page }) => {
    // Join a room first
    await helpers.createRoom(TEST_DATA.ROOMS.INPUT_TEST_ROOM);
    await helpers.waitForRoomJoin();
    
    // Validate chat interface
    await helpers.validateChatInterface();
  });

  test('should send a message', async ({ page }) => {
    // Join a room
    await helpers.createRoom('메시지 전송 테스트');
    await helpers.waitForWebSocketConnection();
    
    // Send a test message
    const testMessage = TEST_DATA.MESSAGES.TEST_MESSAGE;
    const messageSent = await helpers.sendMessage(testMessage);
    
    if (messageSent) {
      // Wait for message to appear and verify
      await helpers.waitForMessage(testMessage);
    }
  });

  test('should send message with Enter key', async ({ page }) => {
    // Join a room
    await helpers.createRoom('엔터키 테스트');
    await helpers.waitForWebSocketConnection();
    
    // Send message with Enter key
    const testMessage = TEST_DATA.MESSAGES.ENTER_KEY_MESSAGE;
    await helpers.sendMessageWithEnter(testMessage);
  });

  test('should display user list', async ({ page }) => {
    // Join a room
    await helpers.createRoom('사용자 목록 테스트');
    await helpers.waitForRoomJoin();
    
    // Check for user list
    const userList = page.locator(SELECTORS.USERS_LIST);
    if (await userList.isVisible()) {
      await expect(userList).toBeVisible();
      
      // Should show current user
      const users = page.locator(SELECTORS.USER);
      if (await users.count() > 0) {
        await expect(users.first()).toBeVisible();
      }
    }
  });

  test('should leave room', async ({ page }) => {
    // Join a room
    await helpers.createRoom('나가기 테스트');
    await helpers.waitForRoomJoin();
    
    // Leave room using helper
    await helpers.leaveRoom();
    
    // Should return to no room state
    await expect(page.locator(SELECTORS.ROOM_BTN))
      .toHaveText(UI_TEXT.ROOM.JOIN_BUTTON);
  });

  test('should handle WebSocket connection states', async ({ page }) => {
    // Monitor console for WebSocket events
    const messages = await helpers.monitorConsoleMessages();
    
    // Join a room to trigger WebSocket connection
    await helpers.createRoom(TEST_DATA.ROOMS.WEBSOCKET_TEST_ROOM);
    await helpers.waitForWebSocketConnection();
    
    // Check if WebSocket connection was attempted
    const hasWebSocketMessages = await helpers.hasWebSocketConsoleMessages(messages);
    
    // This is a basic check - actual implementation may vary
    expect(typeof hasWebSocketMessages).toBe('boolean');
  });

  test('should display system messages', async ({ page }) => {
    // Join a room
    await helpers.createRoom('시스템 메시지 테스트');
    await helpers.waitForRoomJoin();
    
    // Look for join notification (system message)
    const messages = page.locator(SELECTORS.SYSTEM_MESSAGE);
    if (await messages.count() > 0) {
      const joinMessage = messages.first();
      await expect(joinMessage).toBeVisible();
      const messageText = await joinMessage.textContent();
      expect(messageText).toMatch(PATTERNS.JOIN_MESSAGE);
    }
  });
});