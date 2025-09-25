# core/models.py
"""Domain models for the RAG system"""
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

@dataclass
class ProcessedDocument:
    """Domain model for documents"""
    id: str
    filename: str
    file_hash: str
    metadata: Dict[str, Any]

@dataclass  
class DocumentChunk:
    """Domain model for document chunks"""
    id: str
    content: str
    document_id: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None

@dataclass
class ChunkSearchResult:
    """Domain model for search results"""
    chunk: DocumentChunk
    score: float
    highlights: Optional[List[str]] = None