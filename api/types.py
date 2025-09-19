# api/types.py
from pydantic import BaseModel
from typing import Optional, List, Dict
from pydantic import BaseModel, HttpUrl


class Message(BaseModel):
    sender: str
    content: str
    
class DocumentsListItem(BaseModel):
    id: str
    filename: str
    download_url: HttpUrl # CHANGED

class ChatRequest(BaseModel):
    question: str

class SearchResult(BaseModel):
    document_name: str
    page_number: int
    content_snippet: str
    document_id: str
    download_url: HttpUrl # ADD THIS LINE

class SearchResponse(BaseModel):
    status: str
    query: str
    results: List[SearchResult]
    total_results: int

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