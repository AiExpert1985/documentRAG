import hashlib
import re
import os
from pathlib import Path

from fastapi import UploadFile, HTTPException

import logging
from config import settings

logger = logging.getLogger(settings.LOGGER_NAME)

def validate_uploaded_file(file: UploadFile) -> None:
    """Validate uploaded file (name, type, size)."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    extension = file.filename.lower().split('.')[-1]
    if extension not in settings.ALLOWED_FILE_EXTENSIONS:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Allowed: {', '.join(settings.ALLOWED_FILE_EXTENSIONS)}"
        )
    
    if file.size and file.size > settings.MAX_FILE_SIZE:
        max_mb = settings.MAX_FILE_SIZE // 1024 // 1024
        raise HTTPException(status_code=413, detail=f"File too large. Max size: {max_mb}MB")

def validate_file_content(file_path: str, filename: str) -> None:
    """Validates that file content matches its extension using magic number verification."""
    extension = Path(filename).suffix.lower()
    
    with open(file_path, 'rb') as f:
        header = f.read(16)
    
    if extension == '.pdf' and not header.startswith(b'%PDF'):
        raise HTTPException(status_code=400, detail="Invalid PDF file")
    elif extension in ['.jpg', '.jpeg'] and not (header.startswith(b'\xff\xd8')):
        raise HTTPException(status_code=400, detail="Invalid JPEG file")
    elif extension == '.png' and not header.startswith(b'\x89PNG'):
        raise HTTPException(status_code=400, detail="Invalid PNG file")
    
    logger.info(f"Successfully validated file content")

def get_file_hash(content: bytes) -> str:
    """Calculates the SHA256 hash of file content."""
    return hashlib.sha256(content).hexdigest()

def validate_document_id(doc_id: str) -> bool:
    """Validate document ID format."""
    uuid_pattern = r'^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$'
    return bool(re.match(uuid_pattern, doc_id, re.IGNORECASE))

def sanitize_filename(filename: str) -> str:
    """Remove dangerous characters from filename."""
    safe_name = re.sub(r'[^\w\-_\.]', '_', filename)
    safe_name = os.path.basename(safe_name)
    return safe_name[:100]

def get_file_extension(filename: str) -> str:
    """Extracts and normalizes the file extension from a filename."""
    return Path(filename).suffix[1:].lower()