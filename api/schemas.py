# api/schemas.py
from pydantic import BaseModel, HttpUrl
from typing import Optional, List, Dict
from enum import Enum

class Message(BaseModel):
    sender: str
    content: str

class DocumentsListItem(BaseModel):
    id: str
    filename: str
    download_url: HttpUrl

class ChatRequest(BaseModel):
    question: str

class SearchResult(BaseModel):
    document_name: str
    page_number: int
    content_snippet: str
    document_id: str
    download_url: HttpUrl

class SearchResponse(BaseModel):
    status: str
    query: str
    results: List[SearchResult]
    total_results: int

# NEW: Custom exception for better error handling
class DocumentProcessingError(Exception):
    """Raised when document processing fails with a specific error code"""
    
    def __init__(self, message: str, error_code: 'ErrorCode'):
        self.message = message
        self.error_code = error_code
        super().__init__(message)
    
    def __str__(self):
        # Format used for logging and progress store
        return f"[{self.error_code.value}] {self.message}"

# NEW: Error codes for clear user feedback
class ErrorCode(str, Enum):
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    INVALID_FORMAT = "INVALID_FORMAT"
    DUPLICATE_FILE = "DUPLICATE_FILE"
    NO_TEXT_FOUND = "NO_TEXT_FOUND"
    PROCESSING_FAILED = "PROCESSING_FAILED"
    OCR_TIMEOUT = "OCR_TIMEOUT"

# NEW: Processing status enum
class ProcessingStatus(str, Enum):
    PENDING = "pending"
    VALIDATING = "validating"
    EXTRACTING_TEXT = "extracting_text"
    GENERATING_EMBEDDINGS = "generating_embeddings"
    STORING = "storing"
    COMPLETED = "completed"
    FAILED = "failed"

class ProcessDocumentResponse(BaseModel):
    status: str
    filename: str
    document_id: str
    chunks: int
    pages: int
    message: Optional[str] = None
    error: Optional[str] = None
    error_code: Optional[ErrorCode] = None

class StatusResponse(BaseModel):
    document_loaded: Optional[str] = None
    chunks_available: int = 0
    ready_for_queries: bool = False

class DocumentsListResponse(BaseModel):
    documents: List[DocumentsListItem]

class DeleteResponse(BaseModel):
    status: str
    message: str

# NEW: Progress tracking response
class ProcessingProgress(BaseModel):
    document_id: str
    filename: str
    status: ProcessingStatus
    progress_percent: int  # 0-100
    current_step: str
    error: Optional[str] = None
    error_code: Optional[ErrorCode] = None