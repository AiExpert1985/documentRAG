# utils/helpers.py
"""Utility helper functions"""
import hashlib
import re

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

def validate_uploaded_file(file: UploadFile) -> None:
    """Complete file validation."""
    validate_file_exists(file)
    validate_file_type(file)
    validate_file_size(file)

def get_file_hash(content: bytes) -> str:
    """Calculates the SHA256 hash of file content."""
    return hashlib.sha256(content).hexdigest()


# Utility functions
def validate_document_id(doc_id: str) -> bool:
    uuid_pattern = r'^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$'
    return bool(re.match(uuid_pattern, doc_id, re.IGNORECASE))