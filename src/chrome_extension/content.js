/**
 * Content script for the YouTube RAG Chrome extension.
 * This script is injected into YouTube video pages.
 */

// Configuration
const API_URL = 'http://localhost:8000';
let videoId = '';
let videoUrl = '';
let isWidgetOpen = false;
let isProcessing = false;
let processingStatus = null;
let queryHistory = [];

// Create and inject the widget
function createWidget() {
  // Extract video ID from URL
  const urlParams = new URLSearchParams(window.location.search);
  videoId = urlParams.get('v');
  videoUrl = window.location.href;
  
  if (!videoId) {
    console.error('Could not extract video ID from URL');
    return;
  }
  
  // Create widget container
  const widgetContainer = document.createElement('div');
  widgetContainer.id = 'yt-rag-widget-container';
  widgetContainer.className = 'yt-rag-widget-container';
  
  // Create widget toggle button
  const toggleButton = document.createElement('button');
  toggleButton.id = 'yt-rag-toggle-button';
  toggleButton.className = 'yt-rag-toggle-button';
  toggleButton.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"></path><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>';
  toggleButton.title = 'YouTube RAG Assistant';
  
  // Create widget content
  const widgetContent = document.createElement('div');
  widgetContent.id = 'yt-rag-widget-content';
  widgetContent.className = 'yt-rag-widget-content';
  widgetContent.style.display = 'none';
  
  // Add header
  const header = document.createElement('div');
  header.className = 'yt-rag-header';
  header.innerHTML = `
    <h2>YouTube RAG Assistant</h2>
    <button id="yt-rag-close-button" class="yt-rag-close-button">×</button>
  `;
  
  // Add content
  const content = document.createElement('div');
  content.className = 'yt-rag-content';
  content.innerHTML = `
    <div id="yt-rag-status" class="yt-rag-status">
      <p>Process this video to ask questions about its content.</p>
      <button id="yt-rag-process-button" class="yt-rag-button">Process Video</button>
    </div>
    <div id="yt-rag-query-container" class="yt-rag-query-container" style="display: none;">
      <input type="text" id="yt-rag-query-input" class="yt-rag-query-input" placeholder="Ask a question about this video...">
      <button id="yt-rag-query-button" class="yt-rag-button">Ask</button>
    </div>
    <div id="yt-rag-result-container" class="yt-rag-result-container"></div>
    <div id="yt-rag-history-container" class="yt-rag-history-container">
      <h3>Recent Questions</h3>
      <ul id="yt-rag-history-list" class="yt-rag-history-list"></ul>
    </div>
  `;
  
  // Assemble widget
  widgetContent.appendChild(header);
  widgetContent.appendChild(content);
  widgetContainer.appendChild(toggleButton);
  widgetContainer.appendChild(widgetContent);
  
  // Add widget to page
  document.body.appendChild(widgetContainer);
  
  // Add event listeners
  toggleButton.addEventListener('click', toggleWidget);
  document.getElementById('yt-rag-close-button').addEventListener('click', toggleWidget);
  document.getElementById('yt-rag-process-button').addEventListener('click', processVideo);
  document.getElementById('yt-rag-query-button').addEventListener('click', submitQuery);
  document.getElementById('yt-rag-query-input').addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
      submitQuery();
    }
  });
  
  // Check if video is already processed
  checkVideoStatus();
}

// Toggle widget visibility
function toggleWidget() {
  const widgetContent = document.getElementById('yt-rag-widget-content');
  isWidgetOpen = !isWidgetOpen;
  
  if (isWidgetOpen) {
    widgetContent.style.display = 'block';
    // Update processing status if needed
    if (isProcessing) {
      updateProcessingStatus();
    }
  } else {
    widgetContent.style.display = 'none';
  }
}

// Check if video is already processed
async function checkVideoStatus() {
  try {
    const response = await fetch(`${API_URL}/processing-status/${videoId}`);
    
    if (response.ok) {
      const data = await response.json();
      processingStatus = data;
      
      if (data.status === 'completed') {
        showQueryInterface();
      } else if (data.status === 'processing') {
        showProcessingStatus(data);
        isProcessing = true;
        updateProcessingStatus();
      }
    } else if (response.status === 404) {
      // Video not processed yet, show process button
      showProcessButton();
    } else {
      console.error('Error checking video status:', response.statusText);
    }
  } catch (error) {
    console.error('Error checking video status:', error);
  }
}

// Process video
async function processVideo() {
  try {
    const statusElement = document.getElementById('yt-rag-status');
    statusElement.innerHTML = '<p>Starting processing...</p><div class="yt-rag-loader"></div>';
    
    const response = await fetch(`${API_URL}/process-video`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        youtube_url: videoUrl,
        force_reprocess: false
      })
    });
    
    if (response.ok) {
      const data = await response.json();
      processingStatus = data;
      isProcessing = true;
      updateProcessingStatus();
    } else {
      const errorData = await response.json();
      statusElement.innerHTML = `<p>Error: ${errorData.detail || 'Unknown error'}</p><button id="yt-rag-process-button" class="yt-rag-button">Retry</button>`;
      document.getElementById('yt-rag-process-button').addEventListener('click', processVideo);
    }
  } catch (error) {
    const statusElement = document.getElementById('yt-rag-status');
    statusElement.innerHTML = `<p>Error: ${error.message}</p><button id="yt-rag-process-button" class="yt-rag-button">Retry</button>`;
    document.getElementById('yt-rag-process-button').addEventListener('click', processVideo);
  }
}

// Update processing status
async function updateProcessingStatus() {
  if (!isProcessing || !isWidgetOpen) return;
  
  try {
    const response = await fetch(`${API_URL}/processing-status/${videoId}`);
    
    if (response.ok) {
      const data = await response.json();
      processingStatus = data;
      
      if (data.status === 'completed') {
        isProcessing = false;
        showQueryInterface();
      } else if (data.status === 'processing') {
        showProcessingStatus(data);
        // Check again in 2 seconds
        setTimeout(updateProcessingStatus, 2000);
      } else if (data.status === 'error') {
        isProcessing = false;
        const statusElement = document.getElementById('yt-rag-status');
        statusElement.innerHTML = `<p>Error: ${data.message || 'Unknown error'}</p><button id="yt-rag-process-button" class="yt-rag-button">Retry</button>`;
        document.getElementById('yt-rag-process-button').addEventListener('click', processVideo);
      }
    } else {
      console.error('Error updating processing status:', response.statusText);
      setTimeout(updateProcessingStatus, 5000);
    }
  } catch (error) {
    console.error('Error updating processing status:', error);
    setTimeout(updateProcessingStatus, 5000);
  }
}

// Show processing status
function showProcessingStatus(status) {
  const statusElement = document.getElementById('yt-rag-status');
  const progress = Math.round(status.progress * 100);
  
  statusElement.innerHTML = `
    <p>${status.message || 'Processing video...'}</p>
    <div class="yt-rag-progress-container">
      <div class="yt-rag-progress-bar" style="width: ${progress}%"></div>
    </div>
    <p>${progress}% complete</p>
  `;
}

// Show process button
function showProcessButton() {
  const statusElement = document.getElementById('yt-rag-status');
  statusElement.innerHTML = `
    <p>Process this video to ask questions about its content.</p>
    <button id="yt-rag-process-button" class="yt-rag-button">Process Video</button>
  `;
  document.getElementById('yt-rag-process-button').addEventListener('click', processVideo);
}

// Show query interface
function showQueryInterface() {
  const statusElement = document.getElementById('yt-rag-status');
  statusElement.innerHTML = '<p>Video processed successfully! Ask questions below.</p>';
  
  const queryContainer = document.getElementById('yt-rag-query-container');
  queryContainer.style.display = 'flex';
  
  // Load query history
  loadQueryHistory();
}

// Submit query
async function submitQuery() {
  const queryInput = document.getElementById('yt-rag-query-input');
  const query = queryInput.value.trim();
  
  if (!query) return;
  
  const resultContainer = document.getElementById('yt-rag-result-container');
  resultContainer.innerHTML = '<div class="yt-rag-loader"></div><p>Generating answer...</p>';
  
  try {
    const response = await fetch(`${API_URL}/query`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        youtube_url: videoUrl,
        query: query
      })
    });
    
    if (response.ok) {
      const data = await response.json();
      displayResult(data);
      
      // Add to history
      addToHistory(query, data);
      
      // Clear input
      queryInput.value = '';
    } else {
      const errorData = await response.json();
      resultContainer.innerHTML = `<p class="yt-rag-error">Error: ${errorData.detail || 'Unknown error'}</p>`;
    }
  } catch (error) {
    resultContainer.innerHTML = `<p class="yt-rag-error">Error: ${error.message}</p>`;
  }
}

// Display query result
function displayResult(result) {
  const resultContainer = document.getElementById('yt-rag-result-container');
  
  // Format answer with citation links
  let formattedAnswer = result.answer;
  
  // Replace timestamp citations [MM:SS] with links
  formattedAnswer = formattedAnswer.replace(/\[(\d{2}):(\d{2})\]/g, (match, minutes, seconds) => {
    const totalSeconds = parseInt(minutes) * 60 + parseInt(seconds);
    return `<a href="https://www.youtube.com/watch?v=${videoId}&t=${totalSeconds}s" target="_blank" class="yt-rag-timestamp">${match}</a>`;
  });
  
  // Create result HTML
  resultContainer.innerHTML = `
    <div class="yt-rag-answer">
      <h3>Answer</h3>
      <p>${formattedAnswer}</p>
    </div>
    <div class="yt-rag-sources">
      <h3>Sources</h3>
      <div class="yt-rag-sources-list">
        ${result.sources.map((source, index) => `
          <div class="yt-rag-source">
            <h4>Source ${index + 1} - <a href="https://www.youtube.com/watch?v=${videoId}&t=${Math.floor(source.timestamp)}s" target="_blank">${source.timestamp_str}</a></h4>
            <p>${source.content}</p>
          </div>
        `).join('')}
      </div>
    </div>
  `;
  
  // Add event listeners to timestamp links
  const timestampLinks = resultContainer.querySelectorAll('.yt-rag-timestamp');
  timestampLinks.forEach(link => {
    link.addEventListener('click', (e) => {
      e.preventDefault();
      const url = new URL(link.href);
      const time = url.searchParams.get('t');
      if (time) {
        // Seek video to timestamp
        const videoElement = document.querySelector('video');
        if (videoElement) {
          videoElement.currentTime = parseInt(time);
          videoElement.play();
        } else {
          // If can't control video directly, just navigate
          window.location.href = link.href;
        }
      }
    });
  });
}

// Add query to history
function addToHistory(query, result) {
  const historyItem = {
    query,
    timestamp: new Date().toISOString(),
    result
  };
  
  // Add to beginning of array
  queryHistory.unshift(historyItem);
  
  // Limit history to 10 items
  if (queryHistory.length > 10) {
    queryHistory.pop();
  }
  
  // Save to storage
  chrome.storage.local.set({ [`yt-rag-history-${videoId}`]: queryHistory });
  
  // Update history display
  updateHistoryDisplay();
}

// Load query history
function loadQueryHistory() {
  chrome.storage.local.get([`yt-rag-history-${videoId}`], (result) => {
    queryHistory = result[`yt-rag-history-${videoId}`] || [];
    updateHistoryDisplay();
  });
}

// Update history display
function updateHistoryDisplay() {
  const historyList = document.getElementById('yt-rag-history-list');
  
  if (queryHistory.length === 0) {
    historyList.innerHTML = '<li class="yt-rag-history-empty">No questions yet</li>';
    return;
  }
  
  historyList.innerHTML = queryHistory.map((item, index) => `
    <li class="yt-rag-history-item" data-index="${index}">
      <span class="yt-rag-history-query">${item.query}</span>
      <span class="yt-rag-history-time">${formatTimestamp(item.timestamp)}</span>
    </li>
  `).join('');
  
  // Add event listeners
  const historyItems = historyList.querySelectorAll('.yt-rag-history-item');
  historyItems.forEach(item => {
    item.addEventListener('click', () => {
      const index = parseInt(item.dataset.index);
      const historyItem = queryHistory[index];
      
      // Fill query input
      document.getElementById('yt-rag-query-input').value = historyItem.query;
      
      // Display result
      displayResult(historyItem.result);
    });
  });
}

// Format timestamp
function formatTimestamp(timestamp) {
  const date = new Date(timestamp);
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

// Initialize when DOM is loaded
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', createWidget);
} else {
  createWidget();
}

// Listen for URL changes (for YouTube's SPA navigation)
let lastUrl = location.href;
new MutationObserver(() => {
  const url = location.href;
  if (url !== lastUrl) {
    lastUrl = url;
    // Check if we're on a video page
    if (url.includes('youtube.com/watch?')) {
      // Remove existing widget if any
      const existingWidget = document.getElementById('yt-rag-widget-container');
      if (existingWidget) {
        existingWidget.remove();
      }
      // Create new widget
      createWidget();
    }
  }
}).observe(document, { subtree: true, childList: true });
