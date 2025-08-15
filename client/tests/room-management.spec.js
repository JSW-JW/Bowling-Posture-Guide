// tests/room-management.spec.js
import { test, expect } from '@playwright/test';
import { TestHelpers } from './helpers.js';
import { UI_TEXT, SELECTORS, TEST_DATA } from './constants.js';

test.describe('Room Management Features', () => {
  let helpers;

  test.beforeEach(async ({ page }) => {
    helpers = new TestHelpers(page);
    await helpers.navigateToApp();
  });

  test('should open room selection modal', async ({ page }) => {
    await helpers.openRoomSelection();
    
    // Room selection modal should appear with correct title
    await expect(page.locator(SELECTORS.ROOM_SELECTION_TITLE))
      .toHaveText(UI_TEXT.ROOM.MODAL_TITLE);
  });

  test('should display existing rooms list', async ({ page }) => {
    await helpers.openRoomSelection();
    
    // Check if rooms list is visible (even if empty)
    await expect(page.locator(SELECTORS.EXISTING_ROOMS)).toBeVisible();
    await expect(page.locator(SELECTORS.EXISTING_ROOMS_TITLE))
      .toHaveText(UI_TEXT.ROOM.EXISTING_ROOMS);
    
    // Should show loading or empty state initially
    const roomsList = page.locator(SELECTORS.ROOMS_LIST);
    await expect(roomsList).toBeVisible();
  });

  test('should have create new room form', async ({ page }) => {
    await helpers.openRoomSelection();
    
    // Check create room section
    await expect(page.locator(SELECTORS.CREATE_ROOM)).toBeVisible();
    await expect(page.locator(SELECTORS.CREATE_ROOM_TITLE))
      .toHaveText(UI_TEXT.ROOM.CREATE_NEW);
    
    // Validate form structure
    await helpers.validateRoomCreationForm();
  });

  test('should validate room creation form', async ({ page }) => {
    await helpers.openRoomSelection();
    
    // Try to submit empty form
    const createBtn = page.locator(SELECTORS.CREATE_SUBMIT_BTN);
    await createBtn.click();
    
    // Should not proceed without room name (browser validation)
    const nameInput = page.locator(SELECTORS.ROOM_NAME_INPUT);
    await expect(nameInput).toBeFocused();
  });

  test('should create a new room successfully', async ({ page }) => {
    const roomName = await helpers.createRoom();
    
    // Verify room was created and joined
    const roomBtn = page.locator(SELECTORS.ROOM_BTN);
    await expect(roomBtn).toHaveText(UI_TEXT.ROOM.JOINED_PREFIX + roomName);
  });

  test('should close modal when clicking close button', async ({ page }) => {
    await helpers.openRoomSelection();
    await helpers.closeModal();
  });

  test('should close modal when clicking outside', async ({ page }) => {
    await helpers.openRoomSelection();
    await helpers.closeModalByBackdrop();
  });

  test('should handle room joining workflow', async ({ page }) => {
    // Create and join a room
    const roomName = TEST_DATA.ROOMS.BOWLING_FEEDBACK_ROOM;
    await helpers.createRoom(roomName, '10');
    
    // Verify we joined the room
    await expect(page.locator(SELECTORS.ROOM_BTN))
      .toHaveText(UI_TEXT.ROOM.JOINED_PREFIX + roomName);
    
    // The chat should now be visible with room info
    const chat = page.locator(SELECTORS.CHAT_CONTAINER);
    if (await chat.isVisible()) {
      await expect(page.locator(SELECTORS.ROOM_INFO)).toBeVisible();
    }
  });

  test('should show loading states', async ({ page }) => {
    await helpers.openRoomSelection();
    
    // Create room form should handle loading state
    await page.locator(SELECTORS.ROOM_NAME_INPUT).fill('로딩 테스트');
    await page.locator(SELECTORS.MAX_USERS_INPUT).fill('3');
    
    const submitBtn = page.locator(SELECTORS.CREATE_SUBMIT_BTN);
    await submitBtn.click();
    
    // Button should show loading or completed state
    const buttonText = await submitBtn.textContent();
    expect(buttonText).toMatch(new RegExp(`(${UI_TEXT.ROOM.CREATE_BUTTON}|${UI_TEXT.ROOM.CREATING_BUTTON})`));
  });

  test('should handle API errors gracefully', async ({ page }) => {
    // Mock network failure for room creation
    await helpers.simulateNetworkError('**/rooms');
    
    await helpers.openRoomSelection();
    await page.locator(SELECTORS.ROOM_NAME_INPUT).fill('에러 테스트');
    await page.locator(SELECTORS.CREATE_SUBMIT_BTN).click();
    
    // Should show some kind of error indication
    await page.waitForTimeout(1000); // Give time for error to show
    
    // Modal should still be visible if error occurred
    await expect(page.locator(SELECTORS.ROOM_SELECTION)).toBeVisible();
    
    // Clean up network mocks
    await helpers.clearNetworkMocks();
  });
});