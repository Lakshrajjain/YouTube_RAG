/**
 * Popup script for the YouTube RAG Chrome extension.
 */

// DOM elements
const statusElement = document.getElementById('status');
const openButton = document.getElementById('open-button');
const apiUrlInput = document.getElementById('api-url');
const saveSettingsButton = document.getElementById('save-settings');

// Default settings
const DEFAULT_API_URL = 'http://localhost:8000';

// Initialize popup
document.addEventListener('DOMContentLoaded', () => {
  // Load settings
  chrome.storage.local.get(['apiUrl'], (result) => {
    apiUrlInput.value = result.apiUrl || DEFAULT_API_URL;
  });
  
  // Check if we're on a YouTube video page
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    const currentTab = tabs[0];
    const url = currentTab.url;
    
    if (url && url.includes('youtube.com/watch?')) {
      // We're on a YouTube video page
      statusElement.textContent = 'You are on a YouTube video page. You can use the assistant.';
      statusElement.classList.add('active');
      statusElement.classList.remove('inactive');
      
      // Show open button
      openButton.style.display = 'block';
      
      // Add click handler for open button
      openButton.addEventListener('click', () => {
        // Send message to content script to open widget
        chrome.tabs.sendMessage(currentTab.id, { action: 'openWidget' });
        window.close();
      });
    } else {
      // Not on a YouTube video page
      statusElement.textContent = 'You are not on a YouTube video page. Navigate to a YouTube video to use the assistant.';
      statusElement.classList.add('inactive');
      statusElement.classList.remove('active');
    }
  });
  
  // Save settings
  saveSettingsButton.addEventListener('click', () => {
    const apiUrl = apiUrlInput.value.trim();
    
    if (!apiUrl) {
      alert('Please enter a valid API URL');
      return;
    }
    
    chrome.storage.local.set({ apiUrl }, () => {
      // Show success message
      saveSettingsButton.textContent = 'Saved!';
      setTimeout(() => {
        saveSettingsButton.textContent = 'Save Settings';
      }, 2000);
    });
  });
});
