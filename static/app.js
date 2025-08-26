const API_BASE = window.location.origin;

async function uploadPDF() {
    const fileInput = document.getElementById('pdfFile');
    const file = fileInput.files[0];
    
    if (!file) {
        alert('Please select a PDF file first');
        return;
    }

    document.getElementById('uploadStatus').innerHTML = '<div class="response">Uploading...</div>';

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
            getStatus();
        }
    } catch (error) {
        document.getElementById('uploadStatus').innerHTML = 
            `<div class="response error-message">Error: ${error.message}</div>`;
    }
}

async function askQuestion() {
    const question = document.getElementById('chatInput').value.trim();
    
    if (!question) {
        alert('Please enter a question');
        return;
    }

    addToChat('You', question, 'user-message');
    document.getElementById('chatInput').value = '';

    try {
        const response = await fetch(`${API_BASE}/query-pdf`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question: question })
        });
        
        const result = await response.json();
        
        if (result.status === 'success') {
            addToChat('AI', result.answer, 'ai-message');
        } else {
            addToChat('Error', result.error, 'error-message');
        }
    } catch (error) {
        addToChat('Error', error.message, 'error-message');
    }
}

async function getStatus() {
    try {
        const response = await fetch(`${API_BASE}/status`);
        const result = await response.json();
        document.getElementById('statusInfo').innerHTML = 
            `<div class="response">${JSON.stringify(result, null, 2)}</div>`;
    } catch (error) {
        document.getElementById('statusInfo').innerHTML = 
            `<div class="response error-message">Error: ${error.message}</div>`;
    }
}

function addToChat(sender, message, cssClass) {
    const chatHistory = document.getElementById('chatHistory');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${cssClass}`;
    messageDiv.innerHTML = `<strong>${sender}:</strong> ${message}`;
    chatHistory.appendChild(messageDiv);
    chatHistory.scrollTop = chatHistory.scrollHeight;
}

document.getElementById('chatInput').addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
        askQuestion();
    }
});

window.onload = function() {
    getStatus();
}