// static/app.js
const API_BASE = window.location.origin;
let isTyping = false;
let loadedDocuments = {};

// --- Modal Logic ---
let onConfirmCallback = null;

function showConfirmationModal(message, onConfirm) {
    onConfirmCallback = onConfirm;
    document.getElementById('modalMessage').textContent = message;
    document.getElementById('confirmationModal').style.display = 'flex';
}

function hideConfirmationModal() {
    document.getElementById('confirmationModal').style.display = 'none';
    onConfirmCallback = null;
}

// --- Main Application Logic ---
document.addEventListener('DOMContentLoaded', () => {
    // --- Event Listeners for main page ---
    document.getElementById('uploadPdfBtn').addEventListener('click', uploadPDF);
    document.getElementById('sendChatBtn').addEventListener('click', askQuestion);
    document.getElementById('chatInput').addEventListener('keypress', handleKeyPress);
    document.getElementById('refreshStatusBtn').addEventListener('click', getStatus);

    document.getElementById('clearChatBtn').addEventListener('click', () => {
        showConfirmationModal(
            "Are you sure you want to permanently delete the chat history?",
            clearChat // Pass the function reference
        );
    });

    document.getElementById('clearAllDocsBtn').addEventListener('click', () => {
        showConfirmationModal(
            "Are you sure you want to clear all loaded documents? This is irreversible.",
            clearAllDocuments // Pass the function reference
        );
    });

    document.getElementById('documentsList').addEventListener('click', (e) => {
        if (e.target.classList.contains('delete-btn')) {
            const docId = e.target.dataset.docId;
            const docName = loadedDocuments[docId] || 'this document';
            showConfirmationModal(
                `Are you sure you want to delete "${docName}"?`,
                () => deleteDocument(docId) // Use an arrow function to pass the ID
            );
        }
    });

    // --- Event Listeners for Modal ---
    document.getElementById('confirmBtn').addEventListener('click', () => {
        if (onConfirmCallback) {
            onConfirmCallback();
        }
        hideConfirmationModal();
    });
    document.getElementById('cancelBtn').addEventListener('click', hideConfirmationModal);
    document.getElementById('confirmationModal').addEventListener('click', (e) => {
        if (e.target.id === 'confirmationModal') hideConfirmationModal();
    });

    // Initial state load
    checkConnection();
    getStatus();
    listDocuments();
    loadChatHistory();
    setInterval(() => !isTyping && checkConnection(), 30000);
});

async function checkConnection() {
    try {
        const response = await fetch(`${API_BASE}/status`);
        updateConnectionStatus(response.ok);
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
        addToChat('System', 'Please select a PDF file first.', 'system-message error');
        return;
    }

    const uploadButton = document.getElementById('uploadPdfBtn');
    const originalText = uploadButton.textContent;
    uploadButton.innerHTML = '<span class="loading"></span> Processing...';
    uploadButton.disabled = true;

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch(`${API_BASE}/upload-pdf`, { method: 'POST', body: formData });
        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.detail || 'An unknown error occurred');
        }

        addToChat('System', `✅ ${result.message}`, 'system-message success');
        getStatus();
        listDocuments();
        fileInput.value = '';
        document.getElementById('chatInput').focus();

    } catch (error) {
        addToChat('System', `❌ Upload failed: ${error.message}`, 'system-message error');
    } finally {
        uploadButton.textContent = originalText;
        uploadButton.disabled = false;
    }
}

async function askQuestion() {
    if (isTyping) return;
    const chatInput = document.getElementById('chatInput');
    const question = chatInput.value.trim();
    if (!question) return;

    addToChat('You', question, 'user-message');
    chatInput.value = '';

    isTyping = true;
    const typingIndicator = addToChat('AI', '💭 Searching...', 'ai-message typing');

    try {
        const response = await fetch(`${API_BASE}/search`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question: question })
        });

        const result = await response.json();
        if (typingIndicator) typingIndicator.remove();

        if (!response.ok) {
            throw new Error(result.detail || 'An unknown error occurred');
        }

        if (result.results && result.results.length > 0) {
            addToChat('AI', `I found ${result.total_results} relevant snippets for your query:`, 'ai-message');

            result.results.forEach(chunk => {
                let message = `📄 **From: ${chunk.document_name} (Page ${chunk.page_number})**\n\n`;
                message += `"${chunk.content_snippet}"`;
                addToChat('AI', message, 'ai-message');
            });

        } else {
            addToChat('AI', "I couldn't find any relevant snippets for your query in the loaded documents.", 'ai-message');
        }

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

        let statusHtml = `<div><strong>Document(s) Loaded:</strong> ${result.document_loaded || 'None'}</div>`;
        statusHtml += `<div><strong>Chunks Available:</strong> ${result.chunks_available}</div>`;
        statusHtml += `<div><strong>Ready for Queries:</strong> ${result.ready_for_queries ? 'Yes' : 'No'}</div>`;
        document.getElementById('statusInfo').innerHTML = statusHtml;

        updateDocumentIndicator(result);
        updateConnectionStatus(true);
    } catch (error) {
        document.getElementById('statusInfo').innerHTML = `<div class="error">Error fetching status: ${error.message}</div>`;
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
                li.innerHTML = `<span>${doc.filename}</span><button class="delete-btn" data-doc-id="${doc.id}">Delete</button>`;
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

async function loadChatHistory() {
    try {
        const response = await fetch(`${API_BASE}/search-history`);
        if (response.ok) {
            const history = await response.json();
            history.forEach(searchEvent => {
                if (searchEvent.query) {
                    addToChat(
                        'You',
                        searchEvent.query,
                        'user-message'
                    );
                }
            });
        }
    } catch (error) {
        console.error("Failed to load chat history:", error);
    }
}

function updateDocumentIndicator(status) {
    const indicator = document.getElementById('documentIndicator');
    const docCount = status.document_loaded ? status.document_loaded.split(',').length : 0;
    const plural = docCount !== 1 ? 's' : '';

    if (status.ready_for_queries) {
        indicator.textContent = `📄 ${docCount} Document${plural} Loaded`;
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

function isArabic(text) {
    const arabicRegex = /[\u0600-\u06FF]/;
    return arabicRegex.test(text);
}

function addToChat(sender, message, cssClass) {
    const chatHistory = document.getElementById('chatHistory');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${cssClass}`;

    if (isArabic(message)) {
        messageDiv.classList.add('rtl');
    }

    const sanitizedMessage = message.replace(/</g, "&lt;").replace(/>/g, "&gt;");

    if (sender === 'System') {
        messageDiv.innerHTML = `<div class="system-content">${sanitizedMessage.replace(/\n/g, '<br>')}</div>`;
    } else {
        messageDiv.innerHTML = `
            <div class="message-header">
                <strong>${sender}</strong>
                <span class="timestamp">${new Date().toLocaleTimeString()}</span>
            </div>
            <div class="message-content">${sanitizedMessage.replace(/\n/g, '<br>')}</div>
        `;
    }

    chatHistory.appendChild(messageDiv);
    chatHistory.scrollTop = chatHistory.scrollHeight;
    return messageDiv;
}

async function deleteDocument(documentId) {
    const docName = loadedDocuments[documentId] || 'document';
    addToChat('System', `Deleting "${docName}"...`, 'system-message info');
    try {
        const response = await fetch(`${API_BASE}/documents/${documentId}`, { method: 'DELETE' });
        if (!response.ok) {
            const errorResult = await response.json();
            throw new Error(errorResult.detail || 'Failed to delete document');
        }
        addToChat('System', `✅ Document "${docName}" deleted successfully.`, 'system-message success');
    } catch (error) {
        addToChat('System', `❌ Error deleting document: ${error.message}`, 'system-message error');
    } finally {
        getStatus();
        listDocuments();
    }
}

async function clearAllDocuments() {
    addToChat('System', `Clearing all documents...`, 'system-message info');
    try {
        const response = await fetch(`${API_BASE}/documents`, { method: 'DELETE' });
        if (!response.ok) {
            const errorResult = await response.json();
            throw new Error(errorResult.detail || 'Failed to clear documents');
        }
        addToChat('System', `✅ All documents cleared successfully.`, 'system-message success');
    } catch (error) {
        addToChat('System', `❌ Error clearing documents: ${error.message}`, 'system-message error');
    } finally {
        getStatus();
        listDocuments();
    }
}

async function clearChat() {
    try {
        const response = await fetch(`${API_BASE}/search-history`, { method: 'DELETE' });
        if (!response.ok) {
            const errorResult = await response.json();
            throw new Error(errorResult.detail || 'Failed to clear chat history.');
        }
        document.getElementById('chatHistory').innerHTML = '';
        addToChat('System', '✅ Chat history cleared.', 'system-message success');
    } catch (error) {
        addToChat('System', `❌ Error: ${error.message}`, 'system-message error');
    }
}