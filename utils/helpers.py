# utils/helpers.py

import hashlib
import re
import os
from pathlib import Path

from fastapi import UploadFile, HTTPException

import logging
from config import settings

logger = logging.getLogger(__name__)

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

def validate_file_content(file_path: str, filename: str) -> None:
    """
    Validates that file content matches its extension using magic number verification.
    
    Reads the first 16 bytes of the file to check magic numbers (file signatures)
    and ensures they match the declared file extension. This prevents malicious
    files from bypassing extension-based validation (e.g., a .exe renamed to .pdf).
    
    Args:
        file_path: Full path to the saved file on disk
        filename: Original filename to extract extension from
        
    Returns:
        None: Validation passes silently
        
    Raises:
        HTTPException: If file content doesn't match extension
            - 400: Invalid PDF file (missing %PDF header)
            - 400: Invalid JPEG file (missing FF D8 header)
            - 400: Invalid PNG file (missing PNG signature)
            
    Example:
        validate_file_content("/uploads/abc.pdf", "document.pdf")
        # Passes if file starts with %PDF
        # Raises HTTPException if it's actually a .jpg renamed to .pdf
        
    Note:
        This is a security measure that complements extension validation.
        Must be called AFTER file is saved to disk.
    """
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
    
    logger.info(f"✓ Successfully validated file content")

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


def get_file_extension(filename: str) -> str:
    """
    Extracts and normalizes the file extension from a filename.
    
    Removes the leading dot and converts to lowercase to ensure consistent
    extension handling across the application. This prevents issues with
    case-sensitive comparisons (e.g., .PDF vs .pdf).
    
    Args:
        filename: Full filename including extension (e.g., "document.PDF")
        
    Returns:
        str: Lowercase extension without dot (e.g., "pdf")
        
    Example:
        get_file_extension("Report.PDF")  # Returns: "pdf"
        get_file_extension("image.JPG")   # Returns: "jpg"
        get_file_extension("file.txt")    # Returns: "txt"
        
    Note:
        This is the single source of truth for file extension normalization.
        All file type checks should use this function to avoid inconsistent
        case handling throughout the codebase.
    """
    return Path(filename).suffix[1:].lower()