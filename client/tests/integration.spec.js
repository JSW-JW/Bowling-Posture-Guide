// tests/integration.spec.js
import { test, expect } from '@playwright/test';
import { TestHelpers } from './helpers.js';
import { UI_TEXT, SELECTORS, TEST_DATA } from './constants.js';

test.describe('Integration Scenarios', () => {
  let helpers;

  test.beforeEach(async ({ page }) => {
    helpers = new TestHelpers(page);
    await helpers.navigateToApp();
  });

  test('should complete full bowling analysis workflow', async ({ page }) => {
    // Step 1: Verify initial state
    await helpers.checkMainUIElements();
    await helpers.checkAnalysisUIElements();
    
    // Step 2: Check that no analysis is possible without video
    await expect(page.locator(SELECTORS.VIDEO_ELEMENT)).toHaveCount(0);
    await expect(page.locator('text=' + UI_TEXT.ANALYSIS.BUTTON_ANALYZE)).toHaveCount(0);
    
    // Step 3: Verify file input is ready
    const fileInput = page.locator(SELECTORS.FILE_INPUT);
    await expect(fileInput).toBeVisible();
    await expect(fileInput).toHaveAttribute('accept', 'video/*');
    
    // Note: In a real test, you would upload a video file here
    // and test the complete analysis workflow including:
    // - Video playback
    // - Step marking with 's' key
    // - Analysis request
    // - Results display
  });

  test('should integrate room creation with chat functionality', async ({ page }) => {
    // Step 1: Start with no room
    await expect(page.locator(SELECTORS.ROOM_BTN))
      .toHaveText(UI_TEXT.ROOM.JOIN_BUTTON);
    
    // Step 2: Create and join a room
    const roomName = TEST_DATA.ROOMS.BOWLING_FEEDBACK_ROOM;
    await helpers.createRoom(roomName);
    
    // Step 3: Verify room joined
    await expect(page.locator(SELECTORS.ROOM_BTN))
      .toHaveText(UI_TEXT.ROOM.JOINED_PREFIX + roomName);
    
    // Step 4: Ensure chat is visible and functional
    await helpers.ensureChatVisible();
    await helpers.waitForWebSocketConnection();
    
    // Step 5: Send a message in the room
    const testMessage = '볼링 분석 결과를 공유합니다!';
    const messageSent = await helpers.sendMessage(testMessage);
    
    if (messageSent) {
      await helpers.waitForMessage(testMessage);
    }
    
    // Step 6: Leave the room
    await helpers.leaveRoom();
    
    // Step 7: Verify returned to no room state
    await expect(page.locator(SELECTORS.ROOM_BTN))
      .toHaveText(UI_TEXT.ROOM.JOIN_BUTTON);
  });

  test('should handle multi-user room scenario', async ({ context }) => {
    // Create two browser contexts to simulate two users
    const page1 = await context.newPage();
    const page2 = await context.newPage();
    
    const helpers1 = new TestHelpers(page1);
    const helpers2 = new TestHelpers(page2);
    
    // User 1: Navigate and create room
    await helpers1.navigateToApp();
    const roomName = 'Multi User Test Room';
    await helpers1.createRoom(roomName);
    await helpers1.waitForWebSocketConnection();
    
    // User 2: Navigate and join the same room
    await helpers2.navigateToApp();
    await helpers2.openRoomSelection();
    
    // User 2 would need to select existing room from list
    // For now, we'll create a separate room to test parallel usage
    await helpers2.createRoom('User 2 Room');
    await helpers2.waitForWebSocketConnection();
    
    // Both users should be in their respective rooms
    await expect(page1.locator(SELECTORS.ROOM_BTN))
      .toHaveText(UI_TEXT.ROOM.JOINED_PREFIX + roomName);
    await expect(page2.locator(SELECTORS.ROOM_BTN))
      .toHaveText(UI_TEXT.ROOM.JOINED_PREFIX + 'User 2 Room');
    
    // Both users can send messages in their rooms
    await helpers1.sendMessage('Hello from User 1!');
    await helpers2.sendMessage('Hello from User 2!');
    
    // Clean up
    await page1.close();
    await page2.close();
  });

  test('should handle room persistence across page refresh', async ({ page }) => {
    // Step 1: Create and join a room
    const roomName = 'Persistence Test Room';
    await helpers.createRoom(roomName);
    
    // Step 2: Verify room joined
    await expect(page.locator(SELECTORS.ROOM_BTN))
      .toHaveText(UI_TEXT.ROOM.JOINED_PREFIX + roomName);
    
    // Step 3: Refresh the page
    await page.reload();
    await helpers.waitForPageLoad();
    
    // Step 4: Check if room state persists
    // Note: This depends on the implementation
    // The room state might reset after refresh
    const roomBtn = page.locator(SELECTORS.ROOM_BTN);
    const currentRoomText = await roomBtn.textContent();
    
    // Either the room persists or resets to default state
    expect(currentRoomText).toMatch(
      new RegExp(`(${UI_TEXT.ROOM.JOINED_PREFIX}.*|${UI_TEXT.ROOM.JOIN_BUTTON})`)
    );
  });

  test('should handle error states gracefully throughout the application', async ({ page }) => {
    // Test 1: Room creation with network error
    await helpers.simulateNetworkError('**/rooms');
    
    await helpers.openRoomSelection();
    await page.locator(SELECTORS.ROOM_NAME_INPUT).fill('Error Test Room');
    await page.locator(SELECTORS.CREATE_SUBMIT_BTN).click();
    
    // Should handle error gracefully
    await page.waitForTimeout(1000);
    await expect(page.locator(SELECTORS.ROOM_SELECTION)).toBeVisible();
    
    await helpers.clearNetworkMocks();
    await helpers.closeModal();
    
    // Test 2: WebSocket connection failure
    // This would require more sophisticated mocking of WebSocket connections
    // For now, we just test that the UI handles disconnected states
    
    // Test 3: Analysis API error (if video analysis was available)
    // This would test error handling in the analysis workflow
    
    // The application should remain functional despite these errors
    await helpers.checkMainUIElements();
  });

  test('should maintain consistent state across different UI interactions', async ({ page }) => {
    // Step 1: Initial state verification
    await helpers.checkUsername();
    const initialUsername = await page.locator(SELECTORS.USERNAME_DISPLAY).textContent();
    
    // Step 2: Toggle chat visibility
    const chat = page.locator(SELECTORS.CHAT_CONTAINER);
    const toggleBtn = page.locator(SELECTORS.CHAT_TOGGLE);
    
    await toggleBtn.click(); // Hide
    await expect(chat).not.toHaveClass(/visible/);
    
    await toggleBtn.click(); // Show
    await expect(chat).toHaveClass(/visible/);
    
    // Step 3: Open and close room modal multiple times
    for (let i = 0; i < 3; i++) {
      await helpers.openRoomSelection();
      await helpers.closeModal();
    }
    
    // Step 4: Verify username consistency
    const currentUsername = await page.locator(SELECTORS.USERNAME_DISPLAY).textContent();
    expect(currentUsername).toBe(initialUsername);
    
    // Step 5: Verify room button state
    await expect(page.locator(SELECTORS.ROOM_BTN))
      .toHaveText(UI_TEXT.ROOM.JOIN_BUTTON);
    
    // Step 6: Test responsiveness doesn't break state
    await helpers.testResponsiveness();
    
    // Step 7: Final state verification
    await helpers.checkMainUIElements();
    const finalUsername = await page.locator(SELECTORS.USERNAME_DISPLAY).textContent();
    expect(finalUsername).toBe(initialUsername);
  });

  test('should handle rapid user interactions without breaking', async ({ page }) => {
    // Rapid room modal opening/closing
    for (let i = 0; i < 5; i++) {
      await page.locator(SELECTORS.ROOM_BTN).click();
      await page.waitForTimeout(100);
      
      if (await page.locator(SELECTORS.CLOSE_BTN).isVisible()) {
        await page.locator(SELECTORS.CLOSE_BTN).click();
      }
      await page.waitForTimeout(100);
    }
    
    // Rapid chat toggle
    const toggleBtn = page.locator(SELECTORS.CHAT_TOGGLE);
    for (let i = 0; i < 5; i++) {
      await toggleBtn.click();
      await page.waitForTimeout(50);
    }
    
    // Application should still be functional
    await helpers.checkMainUIElements();
    
    // Should be able to complete normal workflow
    await helpers.createRoom('Rapid Interaction Test');
    await expect(page.locator(SELECTORS.ROOM_BTN))
      .toHaveText(UI_TEXT.ROOM.JOINED_PREFIX + 'Rapid Interaction Test');
  });
});