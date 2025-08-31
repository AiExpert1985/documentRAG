# api/types.py
"""Pydantic schemas for API request/response models"""

from pydantic import BaseModel
from typing import Optional

# Request Schemas
class ChatRequest(BaseModel):
    question: str

class SearchMethodRequest(BaseModel):
    search_method: str  # "semantic", "hybrid", etc.

# Response Schemas  
class ChatResponse(BaseModel):
    status: str
    answer: Optional[str] = None
    error: Optional[str] = None
    document: Optional[str] = None
    chunks_used: Optional[int] = None

class UploadResponse(BaseModel):
    status: str
    filename: Optional[str] = None
    pages: Optional[int] = None
    chunks: Optional[int] = None
    message: Optional[str] = None
    error: Optional[str] = None
    search_method: Optional[str] = None

class StatusResponse(BaseModel):
    current_method: str = "unknown"
    document_loaded: Optional[str] = None
    chunks_available: int = 0
    ready_for_queries: bool = False

class ClearDocumentsResponse(BaseModel):
    status: str
    message: Optional[str] = None
    error: Optional[str] = None