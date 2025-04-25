/**
 * Background script for the YouTube RAG Chrome extension.
 */

// Listen for installation
chrome.runtime.onInstalled.addListener(() => {
  console.log('YouTube RAG Assistant installed');
  
  // Set default settings
  chrome.storage.local.get(['apiUrl'], (result) => {
    if (!result.apiUrl) {
      chrome.storage.local.set({ apiUrl: 'http://localhost:8000' });
    }
  });
});

// Listen for messages from content script
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === 'getApiUrl') {
    chrome.storage.local.get(['apiUrl'], (result) => {
      sendResponse({ apiUrl: result.apiUrl || 'http://localhost:8000' });
    });
    return true; // Required for async response
  }
});
