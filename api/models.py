# api/models.py

from pydantic import BaseModel
from typing import Optional

class ChatRequest(BaseModel):
    prompt: str


class ChatResponse(BaseModel):
    status: str
    answer: Optional[str] = None
    error: Optional[str] = None

class QueryRequest(BaseModel):
    question: str

class LocalPDFRequest(BaseModel):
    pdf_path: str

class StatusResponse(BaseModel):
    document_loaded: Optional[str]
    chunks_available: int
    ready_for_queries: bool