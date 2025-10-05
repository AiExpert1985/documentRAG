# core/enums.py
"""Shared enumerations used across the application."""
from enum import Enum


class ErrorCode(str, Enum):
    """Error codes for user-facing error messages."""
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    INVALID_FORMAT = "INVALID_FORMAT"
    DUPLICATE_FILE = "DUPLICATE_FILE"
    NO_TEXT_FOUND = "NO_TEXT_FOUND"
    PROCESSING_FAILED = "PROCESSING_FAILED"
    OCR_TIMEOUT = "OCR_TIMEOUT"


class ProcessingStatus(str, Enum):
    """Document processing pipeline stages."""
    PENDING = "pending"
    VALIDATING = "validating"
    EXTRACTING_TEXT = "extracting_text"
    GENERATING_EMBEDDINGS = "generating_embeddings"
    STORING = "storing"
    COMPLETED = "completed"
    FAILED = "failed"
    
    @staticmethod
    def from_string(status: str) -> 'ProcessingStatus':
        """Convert string to ProcessingStatus enum."""
        try:
            return ProcessingStatus(status)
        except ValueError:
            return ProcessingStatus.FAILED
        
class DocumentResponseStatus(str, Enum):
    """Status for document upload response."""
    PROCESSING = "processing"  # Upload accepted, processing in background
    ERROR = "error"             # Upload or validation failed