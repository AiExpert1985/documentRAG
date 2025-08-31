# api/types.py
from pydantic import BaseModel
from typing import Optional, List, Dict

class Message(BaseModel):
    sender: str
    content: str
    
class DocumentsListItem(BaseModel):
    id: str
    filename: str

class ChatRequest(BaseModel):
    question: str

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
    retrieval_method: Optional[str] = None

class StatusResponse(BaseModel):
    current_method: str = "unknown"
    document_loaded: Optional[str] = None
    chunks_available: int = 0
    ready_for_queries: bool = False

class DocumentsListResponse(BaseModel):
    documents: List[DocumentsListItem]

class DeleteResponse(BaseModel):
    status: str
    message: Optional[str] = None
    error: Optional[str] = None