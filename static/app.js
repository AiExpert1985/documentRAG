const API_BASE = window.location.origin;
let isTyping = false;

// Initialize the application
window.onload = function () {
    checkConnection();
    getStatus();
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

        if (result.status === 'success') {
            document.getElementById('uploadStatus').innerHTML =
                `<div class="response success">✅ ${result.message}</div>`;

            addToChat('System',
                `PDF "${result.filename}" processed successfully! (${result.pages} pages, ${result.chunks} chunks). You can now ask questions about this document.`,
                'system-message success'
            );

            getStatus();

            // Clear file input
            fileInput.value = '';

            // Focus on chat input
            document.getElementById('chatInput').focus();

        } else {
            document.getElementById('uploadStatus').innerHTML =
                `<div class="response error">❌ Error: ${result.error}</div>`;

            addToChat('System',
                `Upload failed: ${result.error}`,
                'system-message error'
            );
        }

    } catch (error) {
        document.getElementById('uploadStatus').innerHTML =
            `<div class="response error">❌ Connection Error: ${error.message}</div>`;

        addToChat('System',
            `Upload failed due to connection error: ${error.message}`,
            'system-message error'
        );
    } finally {
        uploadButton.textContent = originalText;
        uploadButton.disabled = false;
    }
}

// Unified chat function
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
    const typingIndicator = addToChat('AI', '💭 Thinking...', 'ai-message typing');

    try {
        // Single endpoint call
        const response = await fetch(`${API_BASE}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question: question })
        });

        const result = await response.json();

        // Remove typing indicator
        typingIndicator.remove();

        // Handle different response types
        if (result.status === 'success') {
            let message = result.answer;
            if (result.document && result.chunks_used) {
                message += `\n\n📄 Source: ${result.document} (${result.chunks_used} sections analyzed)`;
            }
            addToChat('AI', message, 'ai-message success');

        } else if (result.status === 'no_document') {
            addToChat('AI',
                `📋 ${result.error}\n\nOnce you upload a document, I'll be able to answer questions about its content!`,
                'ai-message no-document'
            );

            // Highlight upload section
            highlightUploadSection();

        } else {
            addToChat('AI',
                `❌ ${result.error}`,
                'ai-message error'
            );
        }

    } catch (error) {
        // Remove typing indicator
        if (typingIndicator) typingIndicator.remove();

        addToChat('System',
            `🔌 Connection error: ${error.message}`,
            'system-message error'
        );

    } finally {
        isTyping = false;
        // Focus back on input
        document.getElementById('chatInput').focus();
    }
}

// Get system status
async function getStatus() {
    try {
        const response = await fetch(`${API_BASE}/status`);
        const result = await response.json();

        let statusHtml = '<div class="response">';
        statusHtml += `Document Loaded: ${result.document_loaded || 'None'}\n`;
        statusHtml += `Chunks Available: ${result.chunks_available}\n`;
        statusHtml += `Ready for Queries: ${result.ready_for_queries ? 'Yes' : 'No'}`;
        statusHtml += '</div>';

        document.getElementById('statusInfo').innerHTML = statusHtml;

        // Update document indicator
        updateDocumentIndicator(result);

        updateConnectionStatus(true);

    } catch (error) {
        document.getElementById('statusInfo').innerHTML =
            `<div class="response error">Error: ${error.message}</div>`;
        updateConnectionStatus(false);
    }
}

// Update document status indicator
function updateDocumentIndicator(status) {
    const indicator = document.getElementById('documentIndicator');

    if (status.ready_for_queries) {
        indicator.textContent = `📄 ${status.document_loaded} (${status.chunks_available} chunks)`;
        indicator.className = 'document-indicator loaded';
    } else {
        indicator.textContent = '📋 No document loaded';
        indicator.className = 'document-indicator empty';
    }
}

// Highlight upload section
function highlightUploadSection() {
    const uploadSection = document.querySelector('.upload-section');
    uploadSection.classList.add('highlight');

    setTimeout(() => {
        uploadSection.classList.remove('highlight');
    }, 3000);
}

// Add message to chat history
function addToChat(sender, message, cssClass) {
    const chatHistory = document.getElementById('chatHistory');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${cssClass}`;

    if (sender === 'System') {
        messageDiv.innerHTML = `<div class="system-content">${message}</div>`;
    } else {
        const senderClass = sender === 'You' ? 'user-sender' : 'ai-sender';
        messageDiv.innerHTML = `
            <div class="message-header">
                <strong class="${senderClass}">${sender}</strong>
                <span class="timestamp">${new Date().toLocaleTimeString()}</span>
            </div>
            <div class="message-content">${message.replace(/\n/g, '<br>')}</div>
        `;
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

// Clear chat history
function clearChat() {
    const chatHistory = document.getElementById('chatHistory');
    chatHistory.innerHTML = '';
    addToChat('System', 'Chat history cleared. Ready for new questions!', 'system-message');
}