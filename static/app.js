const API_BASE = window.location.origin;
let isTyping = false;
let currentMode = 'general';

// Initialize the application
window.onload = function () {
    checkConnection();
    getStatus();
    updateModeIndicator();
}

// Check server connection
async function checkConnection() {
    try {
        const response = await fetch(`${API_BASE}/status`);
        if (response.ok) {
            updateConnectionStatus(true);
        } else {
            updateConnectionStatus(false);
        }
    } catch (error) {
        updateConnectionStatus(false);
    }
}

function updateConnectionStatus(connected) {
    const indicator = document.getElementById('connectionStatus');
    if (connected) {
        indicator.textContent = 'Connected';
        indicator.className = 'status-indicator connected';
    } else {
        indicator.textContent = 'Disconnected';
        indicator.className = 'status-indicator disconnected';
    }
}

// Upload and process PDF
async function uploadPDF() {
    const fileInput = document.getElementById('pdfFile');
    const file = fileInput.files[0];

    if (!file) {
        alert('Please select a PDF file first');
        return;
    }

    const uploadButton = fileInput.nextElementSibling;
    const originalText = uploadButton.textContent;
    uploadButton.innerHTML = '<span class="loading"></span> Processing...';
    uploadButton.disabled = true;

    document.getElementById('uploadStatus').innerHTML = '<div class="response">Uploading and processing PDF...</div>';

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch(`${API_BASE}/upload-pdf`, {
            method: 'POST',
            body: formData
        });

        const result = await response.json();
        document.getElementById('uploadStatus').innerHTML =
            `<div class="response">${JSON.stringify(result, null, 2)}</div>`;

        if (result.status === 'success') {
            addToChat('System', `PDF "${result.filename}" processed successfully! (${result.pages} pages, ${result.chunks} chunks)`, 'system-message');
            getStatus();
            updateModeIndicator();
        }
    } catch (error) {
        document.getElementById('uploadStatus').innerHTML =
            `<div class="response">Error: ${error.message}</div>`;
    } finally {
        uploadButton.textContent = originalText;
        uploadButton.disabled = false;
    }
}

// Smart chat function that automatically decides between general chat and PDF chat
async function askQuestion() {
    if (isTyping) return;

    const question = document.getElementById('chatInput').value.trim();

    if (!question) {
        alert('Please enter a question');
        return;
    }

    // Add user message to chat
    addToChat('You', question, 'user-message');
    document.getElementById('chatInput').value = '';

    // Set typing state
    isTyping = true;
    const typingIndicator = addToChat('AI', 'Thinking...', 'ai-message');

    try {
        // Check current system status to determine chat mode
        const statusResponse = await fetch(`${API_BASE}/status`);
        const status = await statusResponse.json();

        let endpoint, payload, mode;


        if (status.ready_for_queries) {
            endpoint = '/chat-pdf';  // Changed from '/query-pdf'
            payload = { question: question };
        } else {
            endpoint = '/chat';
            payload = { prompt: question };
        }

        // Update mode if changed
        if (mode !== currentMode) {
            currentMode = mode;
            updateModeIndicator();
        }

        // Make API call
        const response = await fetch(`${API_BASE}${endpoint}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const result = await response.json();

        // Remove typing indicator
        typingIndicator.remove();

        // Add AI response
        if (result.status === 'success') {
            let message = result.answer;
            if (result.document && result.chunks_used) {
                message += `\n\n(Source: ${result.document}, ${result.chunks_used} chunks used)`;
            }
            addToChat('AI', message, 'ai-message');
        } else if (result.status === 'no_pdf') {
            addToChat('AI', result.answer, 'ai-message');
        } else {
            addToChat('Error', result.error, 'error-message');
        }

    } catch (error) {
        // Remove typing indicator
        if (typingIndicator) typingIndicator.remove();
        addToChat('Error', `Connection error: ${error.message}`, 'error-message');
    } finally {
        isTyping = false;
    }
}

// Update mode indicator
function updateModeIndicator() {
    const indicator = document.getElementById('modeIndicator');

    fetch(`${API_BASE}/status`)
        .then(response => response.json())
        .then(status => {
            if (status.ready_for_queries) {
                indicator.textContent = `PDF Mode (${status.document_loaded})`;
                indicator.className = 'mode-indicator pdf';
                currentMode = 'pdf';
            } else {
                indicator.textContent = 'General Mode';
                indicator.className = 'mode-indicator general';
                currentMode = 'general';
            }
        })
        .catch(() => {
            indicator.textContent = 'Unknown Mode';
            indicator.className = 'mode-indicator';
        });
}

// Get system status
async function getStatus() {
    try {
        const response = await fetch(`${API_BASE}/status`);
        const result = await response.json();

        let statusHtml = '<div class="response">';
        statusHtml += `Document Loaded: ${result.document_loaded || 'None'}\n`;
        statusHtml += `Chunks Available: ${result.chunks_available}\n`;
        statusHtml += `Ready for PDF Queries: ${result.ready_for_queries ? 'Yes' : 'No'}`;
        statusHtml += '</div>';

        document.getElementById('statusInfo').innerHTML = statusHtml;
        updateConnectionStatus(true);
    } catch (error) {
        document.getElementById('statusInfo').innerHTML =
            `<div class="response">Error: ${error.message}</div>`;
        updateConnectionStatus(false);
    }
}

// Add message to chat history
function addToChat(sender, message, cssClass) {
    const chatHistory = document.getElementById('chatHistory');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${cssClass}`;

    if (sender === 'System') {
        messageDiv.innerHTML = message;
    } else {
        messageDiv.innerHTML = `<strong>${sender}:</strong><br>${message}`;
    }

    chatHistory.appendChild(messageDiv);
    chatHistory.scrollTop = chatHistory.scrollHeight;

    return messageDiv; // Return element for potential removal
}

// Event listeners
document.getElementById('chatInput').addEventListener('keypress', function (e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        askQuestion();
    }
});

// Auto-refresh status every 30 seconds
setInterval(() => {
    if (!isTyping) {
        checkConnection();
    }
}, 30000);