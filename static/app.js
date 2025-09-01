// app.js
const API_BASE = window.location.origin;
let isTyping = false;
let loadedDocuments = {};

document.addEventListener('DOMContentLoaded', () => {
    checkConnection();
    getStatus();
    listDocuments();
    loadChatHistory();

    // Add event listeners
    document.getElementById('uploadPdfBtn').addEventListener('click', uploadPDF);
    document.getElementById('sendChatBtn').addEventListener('click', askQuestion);
    document.getElementById('chatInput').addEventListener('keypress', handleKeyPress);
    document.getElementById('clearChatBtn').addEventListener('click', clearChat);
    document.getElementById('clearAllDocsBtn').addEventListener('click', clearAllDocuments);
    document.getElementById('refreshStatusBtn').addEventListener('click', getStatus);

    // Use event delegation for the documents list
    document.getElementById('documentsList').addEventListener('click', (e) => {
        if (e.target.classList.contains('delete-btn')) {
            deleteDocument(e.target.dataset.docId);
        }
    });

    // Periodically check connection status
    setInterval(() => {
        if (!isTyping) {
            checkConnection();
        }
    }, 30000);
});

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

async function uploadPDF() {
    const fileInput = document.getElementById('pdfFile');
    const file = fileInput.files[0];

    if (!file) {
        alert('Please select a PDF file first');
        return;
    }

    const uploadButton = document.getElementById('uploadPdfBtn');
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

        if (response.ok) {
            document.getElementById('uploadStatus').innerHTML =
                `<div class="response success">✅ ${result.message}</div>`;

            addToChat('System',
                `PDF "${result.filename}" processed successfully! (${result.pages} pages, ${result.chunks} chunks). You can now ask questions about this document.`,
                'system-message success'
            );

            getStatus();
            listDocuments();
            fileInput.value = '';
            document.getElementById('chatInput').focus();
        } else {
            document.getElementById('uploadStatus').innerHTML =
                `<div class="response error">❌ Error: ${result.detail || 'An unknown error occurred'}</div>`;
            addToChat('System',
                `Upload failed: ${result.detail || 'An unknown error occurred'}`,
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

async function askQuestion() {
    if (isTyping) return;

    const chatInput = document.getElementById('chatInput');
    const question = chatInput.value.trim();

    if (!question) {
        return;
    }

    addToChat('You', question, 'user-message');
    chatInput.value = '';

    isTyping = true;
    const typingIndicator = addToChat('AI', '💭 Thinking...', 'ai-message typing');

    try {
        const response = await fetch(`${API_BASE}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question: question })
        });

        if (!response.ok) {
            const errorResult = await response.json();
            if (typingIndicator) typingIndicator.remove();
            if (response.status === 400) {
                addToChat('AI',
                    `📋 ${errorResult.detail}\n\nOnce you upload a document, I'll be able to answer questions about its content!`,
                    'ai-message no-document'
                );
                highlightUploadSection();
            } else {
                addToChat('AI', `❌ ${errorResult.detail || 'An unknown error occurred'}`, 'ai-message error');
            }
            return;
        }

        const result = await response.json();
        if (typingIndicator) typingIndicator.remove();

        let message = result.answer;
        if (result.document && result.chunks_used) {
            message += `\n\n📄 Sources: ${result.document}`;
        }
        addToChat('AI', message, 'ai-message success');

    } catch (error) {
        if (typingIndicator) typingIndicator.remove();
        addToChat('System', `🔌 Connection error: ${error.message}`, 'system-message error');
    } finally {
        isTyping = false;
        document.getElementById('chatInput').focus();
    }
}

function handleKeyPress(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        askQuestion();
    }
}

async function getStatus() {
    try {
        const response = await fetch(`${API_BASE}/status`);
        const result = await response.json();
        let statusHtml = '<div class="response">';
        statusHtml += `Document(s) Loaded: ${result.document_loaded || 'None'}\n`;
        statusHtml += `Chunks Available: ${result.chunks_available}\n`;
        statusHtml += `Ready for Queries: ${result.ready_for_queries ? 'Yes' : 'No'}`;
        statusHtml += '</div>';
        document.getElementById('statusInfo').innerHTML = statusHtml;
        updateDocumentIndicator(result);
        updateConnectionStatus(true);
    } catch (error) {
        document.getElementById('statusInfo').innerHTML =
            `<div class="response error">Error: ${error.message}</div>`;
        updateConnectionStatus(false);
    }
}

async function listDocuments() {
    try {
        const response = await fetch(`${API_BASE}/documents`);
        const result = await response.json();
        const documentsList = document.getElementById('documentsList');
        documentsList.innerHTML = '';

        loadedDocuments = {};

        if (result.documents && result.documents.length > 0) {
            result.documents.forEach(doc => {
                const li = document.createElement('li');
                li.innerHTML = `
                    <span>${doc.filename}</span>
                    <button class="delete-btn" data-doc-id="${doc.id}">Delete</button>
                `;
                documentsList.appendChild(li);
                loadedDocuments[doc.id] = doc.filename;
            });
        } else {
            documentsList.innerHTML = '<li class="no-docs">No documents loaded.</li>';
        }
    } catch (error) {
        document.getElementById('documentsList').innerHTML = `<li class="error-docs">Error loading documents list.</li>`;
    }
}

async function deleteDocument(documentId) {
    if (!confirm("Are you sure you want to delete this document?")) {
        return;
    }

    addToChat('System', `Deleting document: ${loadedDocuments[documentId]}...`, 'system-message info');

    try {
        const response = await fetch(`${API_BASE}/documents/${documentId}`, {
            method: 'DELETE'
        });
        const result = await response.json();

        if (response.ok) {
            addToChat('System', `✅ Document "${loadedDocuments[documentId]}" deleted successfully.`, 'system-message success');
        } else {
            addToChat('System', `❌ Failed to delete document: ${result.message}`, 'system-message error');
        }

        getStatus();
        listDocuments();
    } catch (error) {
        addToChat('System', `❌ Connection error while deleting document.`, 'system-message error');
    }
}

async function clearAllDocuments() {
    if (!confirm("Are you sure you want to clear all loaded documents? This action is irreversible.")) {
        return;
    }

    addToChat('System', `Clearing all documents...`, 'system-message info');

    try {
        const response = await fetch(`${API_BASE}/documents`, {
            method: 'DELETE'
        });
        const result = await response.json();

        if (response.ok) {
            addToChat('System', `✅ All documents cleared successfully.`, 'system-message success');
        } else {
            addToChat('System', `❌ Failed to clear all documents: ${result.message}`, 'system-message error');
        }

        getStatus();
        listDocuments();
    } catch (error) {
        addToChat('System', `❌ Connection error while clearing documents.`, 'system-message error');
    }
}

async function loadChatHistory() {
    try {
        const response = await fetch(`${API_BASE}/chat-history`);
        if (response.ok) {
            const history = await response.json();
            history.forEach(message => {
                addToChat(
                    message.sender === 'user' ? 'You' : 'AI',
                    message.content,
                    message.sender === 'user' ? 'user-message' : 'ai-message'
                );
            });
        }
    } catch (error) {
        console.error("Failed to load chat history:", error);
    }
}

function updateDocumentIndicator(status) {
    const indicator = document.getElementById('documentIndicator');
    const docCount = status.document_loaded ? status.document_loaded.split(',').length : 0;
    const plural = docCount > 1 ? 's' : '';

    if (status.ready_for_queries) {
        indicator.textContent = `📄 ${docCount} Document${plural} Loaded (${status.chunks_available} chunks)`;
        indicator.className = 'document-indicator loaded';
    } else {
        indicator.textContent = '📋 No document loaded';
        indicator.className = 'document-indicator empty';
    }
}

function highlightUploadSection() {
    const uploadSection = document.querySelector('.upload-section');
    uploadSection.classList.add('highlight');
    setTimeout(() => {
        uploadSection.classList.remove('highlight');
    }, 3000);
}

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

    return messageDiv;
}

async function clearChat() {
    if (!confirm("Are you sure you want to permanently delete the chat history?")) {
        return;
    }

    try {
        const response = await fetch(`${API_BASE}/chat-history`, {
            method: 'DELETE'
        });

        if (!response.ok) {
            const errorResult = await response.json();
            throw new Error(errorResult.detail || 'Failed to clear chat history.');
        }

        // If the backend deletion was successful, now clear the UI
        const chatHistory = document.getElementById('chatHistory');
        chatHistory.innerHTML = '';
        addToChat('System', 'Chat history cleared. Ready for new questions!', 'system-message success');

    } catch (error) {
        console.error("Clear chat error:", error);
        addToChat('System', `❌ Error: ${error.message}`, 'system-message error');
    }
}