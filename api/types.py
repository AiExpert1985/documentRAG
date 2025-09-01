# api/types.py
from pydantic import BaseModel
from typing import Optional, List

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
    answer: str
    document: Optional[str] = None
    chunks_used: Optional[int] = None

class UploadResponse(BaseModel):
    status: str
    filename: str
    pages: int
    chunks: int
    message: str

class StatusResponse(BaseModel):
    document_loaded: Optional[str] = None
    chunks_available: int = 0
    ready_for_queries: bool = False

class DocumentsListResponse(BaseModel):
    documents: List[DocumentsListItem]

class DeleteResponse(BaseModel):
    status: str
    message: str