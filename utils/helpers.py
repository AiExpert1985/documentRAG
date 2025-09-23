# utils/helpers.py - UPDATED with better validation

import hashlib
import re
import os
from pathlib import Path

from fastapi import UploadFile, HTTPException
from config import settings

def validate_file_exists(file: UploadFile) -> None:
    """Validate file has a filename."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

def validate_file_type(file: UploadFile) -> None:
    """Validate file type is supported."""
    file_extension = file.filename.lower().split('.')[-1]
    if file_extension not in settings.ALLOWED_FILE_EXTENSIONS:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Allowed: {', '.join(settings.ALLOWED_FILE_EXTENSIONS)}"
        )

def validate_file_size(file: UploadFile) -> None:
    """Validate file size is within limits."""
    if file.size and file.size > settings.MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413, 
            detail=f"File too large. Max size: {settings.MAX_FILE_SIZE // 1024 // 1024}MB"
        )

# NEW: Add magic number validation for security
def validate_file_content(file_path: str, filename: str) -> None:
    """Validate file content matches extension using magic numbers."""
    extension = Path(filename).suffix.lower()
    
    with open(file_path, 'rb') as f:
        header = f.read(16)  # Read first 16 bytes
    
    # Check magic numbers
    if extension == '.pdf' and not header.startswith(b'%PDF'):
        raise HTTPException(status_code=400, detail="Invalid PDF file")
    elif extension in ['.jpg', '.jpeg'] and not (header.startswith(b'\xff\xd8')):
        raise HTTPException(status_code=400, detail="Invalid JPEG file")
    elif extension == '.png' and not header.startswith(b'\x89PNG'):
        raise HTTPException(status_code=400, detail="Invalid PNG file")

def validate_uploaded_file(file: UploadFile) -> None:
    """Complete file validation."""
    validate_file_exists(file)
    validate_file_type(file)
    validate_file_size(file)

def get_file_hash(content: bytes) -> str:
    """Calculates the SHA256 hash of file content."""
    return hashlib.sha256(content).hexdigest()

def validate_document_id(doc_id: str) -> bool:
    """Validate document ID format."""
    uuid_pattern = r'^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$'
    return bool(re.match(uuid_pattern, doc_id, re.IGNORECASE))

# NEW: Secure filename generation
def sanitize_filename(filename: str) -> str:
    """Remove dangerous characters from filename."""
    # Keep only alphanumeric, dots, dashes, underscores
    safe_name = re.sub(r'[^\w\-_\.]', '_', filename)
    # Prevent directory traversal
    safe_name = os.path.basename(safe_name)
    return safe_name[:100]  # Limit length