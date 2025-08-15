// tests/helpers.js
import { expect } from '@playwright/test';
import { UI_TEXT, SELECTORS, TEST_DATA } from './constants.js';

/**
 * Common setup functions
 */
export class TestHelpers {
  constructor(page) {
    this.page = page;
  }

  // Navigation and setup
  async navigateToApp() {
    await this.page.goto('/');
    await expect(this.page.locator(SELECTORS.MAIN_TITLE)).toHaveText(UI_TEXT.MAIN_TITLE);
  }

  async waitForPageLoad() {
    await expect(this.page.locator(SELECTORS.MAIN_TITLE)).toBeVisible();
  }

  // Room management helpers
  async openRoomSelection() {
    const roomBtn = this.page.locator(SELECTORS.ROOM_BTN);
    await expect(roomBtn).toHaveText(UI_TEXT.ROOM.JOIN_BUTTON);
    await roomBtn.click();
    await expect(this.page.locator(SELECTORS.ROOM_SELECTION)).toBeVisible();
  }

  async createRoom(roomName = TEST_DATA.ROOMS.TEST_ROOM, maxUsers = TEST_DATA.DEFAULT_MAX_USERS_STRING) {
    await this.openRoomSelection();
    
    // Click create room button to show form
    await this.page.locator(SELECTORS.CREATE_ROOM_BTN).click();
    
    // Fill form
    await this.page.locator(SELECTORS.ROOM_NAME_INPUT).fill(roomName);
    
    // Submit
    await this.page.locator(SELECTORS.CREATE_SUBMIT_BTN).click();
    
    // Wait for room join
    await expect(this.page.locator(SELECTORS.ROOM_BTN))
      .toHaveText(UI_TEXT.ROOM.JOINED_PREFIX + roomName);
    
    return roomName;
  }

  async joinRoom(roomName = TEST_DATA.ROOMS.TEST_ROOM) {
    return await this.createRoom(roomName);
  }

  async leaveRoom() {
    const leaveBtn = this.page.locator(SELECTORS.LEAVE_ROOM_BTN);
    if (await leaveBtn.isVisible()) {
      await leaveBtn.click();
      await expect(this.page.locator(SELECTORS.ROOM_BTN))
        .toHaveText(UI_TEXT.ROOM.JOIN_BUTTON);
    }
  }

  // Chat helpers
  async ensureChatVisible() {
    const chat = this.page.locator(SELECTORS.CHAT_CONTAINER);
    if (!(await chat.isVisible()) || !(await chat.evaluate(el => el.classList.contains('visible')))) {
      const toggleBtn = this.page.locator(SELECTORS.CHAT_TOGGLE);
      await toggleBtn.click();
    }
    await expect(chat).toHaveClass(/visible/);
  }

  async sendMessage(message) {
    await this.ensureChatVisible();
    
    const messageInput = this.page.locator(SELECTORS.MESSAGE_INPUT);
    const sendBtn = this.page.locator(SELECTORS.SEND_BTN);
    
    if (await messageInput.isVisible() && await sendBtn.isVisible()) {
      await messageInput.fill(message);
      await sendBtn.click();
      
      // Verify input is cleared
      await expect(messageInput).toHaveValue('');
      return true;
    }
    return false;
  }

  async sendMessageWithEnter(message) {
    await this.ensureChatVisible();
    
    const messageInput = this.page.locator(SELECTORS.MESSAGE_INPUT);
    
    if (await messageInput.isVisible()) {
      await messageInput.fill(message);
      await messageInput.press('Enter');
      
      // Verify input is cleared
      await expect(messageInput).toHaveValue('');
      return true;
    }
    return false;
  }

  async waitForMessage(messageText, timeout = 5000) {
    const message = this.page.locator(SELECTORS.MESSAGE).filter({ hasText: messageText });
    await expect(message).toBeVisible({ timeout });
  }

  // UI interaction helpers
  async checkMainUIElements() {
    // Main title
    await expect(this.page.locator(SELECTORS.MAIN_TITLE)).toBeVisible();
    await expect(this.page.locator(SELECTORS.MAIN_TITLE)).toHaveText(UI_TEXT.MAIN_TITLE);
    
    // File input
    await expect(this.page.locator(SELECTORS.FILE_INPUT)).toBeVisible();
    await expect(this.page.locator(SELECTORS.FILE_INPUT)).toHaveAttribute('accept', 'video/*');
    
    // Controls guide
    await expect(this.page.locator(SELECTORS.CONTROLS_GUIDE)).toBeVisible();
    
    // User info
    await expect(this.page.locator(SELECTORS.USER_INFO)).toBeVisible();
    await expect(this.page.locator(SELECTORS.USERNAME_DISPLAY)).toBeVisible();
    await expect(this.page.locator(SELECTORS.ROOM_BTN)).toBeVisible();
  }

  async checkKeyboardShortcuts() {
    const shortcuts = Object.values(UI_TEXT.SHORTCUTS);
    
    for (const shortcut of shortcuts) {
      await expect(this.page.locator(SELECTORS.KEY_CHIP).filter({ hasText: shortcut }))
        .toBeVisible();
    }
  }

  async checkUsername() {
    const usernameDisplay = this.page.locator(SELECTORS.USERNAME_DISPLAY);
    await expect(usernameDisplay).toBeVisible();
    
    const usernameText = await usernameDisplay.textContent();
    expect(usernameText).toMatch(/👤 볼러\d+_\w+/);
  }

  // Modal helpers
  async closeModal() {
    const closeBtn = this.page.locator(SELECTORS.CLOSE_BTN);
    await closeBtn.click();
    await expect(this.page.locator(SELECTORS.ROOM_SELECTION)).toHaveCount(0);
  }

  async closeModalByBackdrop() {
    const backdrop = this.page.locator(SELECTORS.ROOM_SELECTION);
    await backdrop.click({ position: { x: 10, y: 10 } }); // Click near edge
    await expect(this.page.locator(SELECTORS.ROOM_SELECTION)).toHaveCount(0);
  }

  // Viewport helpers
  async testResponsiveness() {
    const viewports = [
      { width: 1200, height: 800 },
      { width: 768, height: 600 },
      { width: 480, height: 800 }
    ];

    for (const viewport of viewports) {
      await this.page.setViewportSize(viewport);
      await expect(this.page.locator(SELECTORS.MAIN_TITLE)).toBeVisible();
    }
  }

  // Waiting helpers
  async waitForWebSocketConnection(timeout = 3000) {
    await this.page.waitForTimeout(timeout);
  }

  async waitForRoomJoin(timeout = 2000) {
    await this.page.waitForTimeout(timeout);
  }

  // Error testing helpers
  async simulateNetworkError(urlPattern) {
    await this.page.route(urlPattern, route => route.abort());
  }

  async clearNetworkMocks() {
    await this.page.unrouteAll();
  }

  // Console monitoring
  async monitorConsoleMessages() {
    const messages = [];
    this.page.on('console', msg => messages.push(msg.text()));
    return messages;
  }

  async hasWebSocketConsoleMessages(messages) {
    return messages.some(msg => 
      msg.includes('WebSocket') || 
      msg.includes('websocket') || 
      msg.includes('ws')
    );
  }

  // Validation helpers
  async validateRoomCreationForm() {
    // Check that create form button exists
    await expect(this.page.locator(SELECTORS.CREATE_ROOM_BTN)).toBeVisible();
  }

  async validateChatInterface() {
    const chat = this.page.locator(SELECTORS.CHAT_CONTAINER);
    await expect(chat).toBeVisible();
    
    // Check basic chat elements
    const messageInput = this.page.locator(SELECTORS.MESSAGE_INPUT);
    const sendBtn = this.page.locator(SELECTORS.SEND_BTN);
    
    if (await messageInput.isVisible()) {
      await expect(messageInput)
        .toHaveAttribute('placeholder', UI_TEXT.CHAT.MESSAGE_PLACEHOLDER);
    }
    
    if (await sendBtn.isVisible()) {
      await expect(sendBtn).toHaveText(UI_TEXT.CHAT.SEND_BUTTON);
    }
  }

  // Analysis helpers
  async checkAnalysisUIElements() {
    // Initially no video should be visible
    await expect(this.page.locator(SELECTORS.VIDEO_ELEMENT)).toHaveCount(0);
    
    // Status and analysis button should not be visible initially
    await expect(this.page.locator(SELECTORS.STATUS)).toHaveCount(0);
    await expect(this.page.locator(SELECTORS.RESULTS_SECTION)).toHaveCount(0);
  }
}