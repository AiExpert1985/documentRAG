# core/interfaces.py
"""Core interfaces for the RAG system"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from fastapi import UploadFile, Request

from api.types import ProcessDocumentResponse, SearchResult
from core.models import DocumentChunk, ProcessedDocument 

# ============= Vector Store Interface =============
class IVectorStore(ABC):
    """Interface for vector storage operations"""
    
    @abstractmethod
    async def add_chunks(self, chunks: List[DocumentChunk]) -> bool:
        """Add document chunks with embeddings"""
        pass
    
    @abstractmethod
    async def search(self, query_embedding: List[float], top_k: int = 5) -> List[SearchResult]:
        """Search for similar chunks"""
        pass
    
    @abstractmethod
    async def delete_by_document(self, document_id: str) -> bool:
        """Delete all chunks for a document"""
        pass
    
    @abstractmethod
    async def clear(self) -> bool:
        """Clear all stored vectors"""
        pass
    
    @abstractmethod
    async def count(self) -> int:
        """Get total number of chunks"""
        pass

# ============= Document Processor Interface =============
class IDocumentProcessor(ABC):
    """Interface for document processing"""
    
    @abstractmethod
    async def process(self, file_path: str, file_type: str) -> List[DocumentChunk]:
        """Process document and return chunks"""
        pass
    
    @abstractmethod
    async def validate(self, file_path: str, file_type: str) -> bool:
        """Validate if document can be processed"""
        pass

# ============= Embedding Service Interface =============
class IEmbeddingService(ABC):
    """Interface for embedding generation"""
    
    @abstractmethod
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for text chunks"""
        pass
    
    @abstractmethod
    async def generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for search query"""
        pass

# ============= Repository Interfaces =============
class IDocumentRepository(ABC):
    """Interface for document data access"""
    
    @abstractmethod
    async def create(self, document_id: str, filename: str, file_hash: str, stored_filename: str) -> ProcessedDocument:
        """Create new document record"""
        pass
    
    @abstractmethod
    async def get_by_id(self, document_id: str) -> Optional[ProcessedDocument]:
        """Get document by ID"""
        pass
    
    @abstractmethod
    async def get_by_hash(self, file_hash: str) -> Optional[ProcessedDocument]:
        """Check if document with hash exists"""
        pass
    
    @abstractmethod
    async def list_all(self) -> List[ProcessedDocument]:
        """List all documents"""
        pass
    
    @abstractmethod
    async def delete(self, document_id: str) -> bool:
        """Delete document record"""
        pass
    
    @abstractmethod
    async def delete_all(self) -> bool:
        """Delete all documents"""
        pass

class IMessageRepository(ABC):
    """Interface for message/search history"""
    
    @abstractmethod
    async def save_search(self, query: str, results_count: int) -> None:
        """Save search query"""
        pass
    
    @abstractmethod
    async def get_search_history(self, limit: int = 10) -> List[Dict]:
        """Get search history"""
        pass
    
    @abstractmethod
    async def clear_history(self) -> bool:
        """Clear search history"""
        pass

    @abstractmethod
    async def save_search_results(self, query: str, results: List[SearchResult]) -> None:
        """Save search results in structured format"""
        pass

# ============= File Storage Interface =============
class IFileStorage(ABC):
    """Interface for physical file storage operations"""

    @abstractmethod
    async def save(self, file: UploadFile, filename: str) -> str:
        """Save an uploaded file and return its stored name."""
        pass

    @abstractmethod
    async def get_path(self, filename: str) -> Optional[str]:
        """Get the full path to a stored file."""
        pass

    @abstractmethod
    async def delete(self, filename: str) -> bool:
        """Delete a stored file."""
        pass

# ============= Service Layer Interfaces =============
class IRAGService(ABC):
    """High-level RAG operations interface"""
    
    @abstractmethod
    async def process_document(self, file: UploadFile) -> ProcessDocumentResponse:
        """Process and store a document from an UploadFile object."""
        pass

    @abstractmethod
    async def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Search across all documents"""
        pass
    
    @abstractmethod
    async def delete_document(self, document_id: str) -> bool:
        """Delete a document and its chunks"""
        pass
        
    @abstractmethod
    async def clear_all(self) -> bool:
        """Clear all documents and vectors"""
        pass
        
    @abstractmethod
    async def list_documents(self, request: Request) -> List[Dict[str, str]]:
        """List all documents, including download URLs."""
        pass
        
    @abstractmethod
    async def get_document_with_path(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get a document's details and its physical file path."""
        pass

# ============= PDF to Image Converter Interface =============
class IPdfToImageConverter(ABC):
    """Interface for PDF to image conversion"""

    @abstractmethod
    def convert(self, file_path: str) -> List[Any]:
        """Converts a PDF file to a list of images."""
        pass