// tests/bowling-analysis.spec.js
import { test, expect } from '@playwright/test';
import { TestHelpers } from './helpers.js';
import { UI_TEXT, SELECTORS, PATTERNS } from './constants.js';

test.describe('Bowling Analysis Features', () => {
  let helpers;

  test.beforeEach(async ({ page }) => {
    helpers = new TestHelpers(page);
    await helpers.navigateToApp();
  });

  test('should display main UI elements correctly', async ({ page }) => {
    await helpers.checkMainUIElements();
  });

  test('should handle video file upload', async ({ page }) => {
    // Test the file input behavior
    const fileInput = page.locator(SELECTORS.FILE_INPUT);
    await expect(fileInput).toBeVisible();
    await expect(fileInput).toHaveAttribute('accept', 'video/*');
  });

  test('should show username and room controls', async ({ page }) => {
    // Check user info display
    await expect(page.locator(SELECTORS.USER_INFO)).toBeVisible();
    
    // Check username display
    await helpers.checkUsername();
    
    // Check room button
    const roomBtn = page.locator(SELECTORS.ROOM_BTN);
    await expect(roomBtn).toBeVisible();
    await expect(roomBtn).toHaveText(UI_TEXT.ROOM.JOIN_BUTTON);
  });

  test('should show analysis controls when no video is loaded', async ({ page }) => {
    await helpers.checkAnalysisUIElements();
  });

  test('should handle keyboard shortcuts info', async ({ page }) => {
    await helpers.checkKeyboardShortcuts();
  });

  test('should display room join button correctly', async ({ page }) => {
    // Test room button click
    const roomBtn = page.locator(SELECTORS.ROOM_BTN);
    await roomBtn.click();
    
    // This should open the room selection modal (tested in room-management.spec.js)
    // Here we just check that the button works
    await expect(roomBtn).toBeVisible();
  });

  test('should not show analysis results initially', async ({ page }) => {
    // Results section should not be visible initially  
    await expect(page.locator(SELECTORS.RESULTS_SECTION)).toHaveCount(0);
    await expect(page.locator('text=' + UI_TEXT.ANALYSIS.RESULTS_TITLE)).toHaveCount(0);
  });

  test('should show error state for incomplete steps', async ({ page }) => {
    // This test simulates trying to analyze without proper setup
    // Since we can't easily upload a real video in this test, 
    // we check that the UI handles the error case appropriately
    
    // No analysis button should be visible without video upload
    await expect(page.locator('text=' + UI_TEXT.ANALYSIS.BUTTON_ANALYZE)).toHaveCount(0);
    
    // Error message area exists but should be empty initially
    await expect(page.locator(SELECTORS.ERROR_MESSAGE)).toHaveCount(0);
  });

  test('should handle page responsiveness', async ({ page }) => {
    await helpers.testResponsiveness();
  });
});