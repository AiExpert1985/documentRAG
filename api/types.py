# api/schemas.py
"""Pydantic schemas for API request/response models"""

from pydantic import BaseModel
from typing import Optional, List

# Request Schemas
class ChatRequest(BaseModel):
    question: str

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

class StatusResponse(BaseModel):
    document_loaded: Optional[str] = None
    chunks_available: int = 0
    ready_for_queries: bool = False