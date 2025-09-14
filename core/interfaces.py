# core/interfaces.py
"""Core interfaces for the RAG system"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class Document:
    """Domain model for documents"""
    id: str
    filename: str
    file_hash: str
    metadata: Dict[str, Any]

@dataclass
class Chunk:
    """Domain model for document chunks"""
    id: str
    content: str
    document_id: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None

@dataclass
class SearchResult:
    """Domain model for search results"""
    chunk: Chunk
    score: float
    highlights: Optional[List[str]] = None

# ============= Vector Store Interface =============
class IVectorStore(ABC):
    """Interface for vector storage operations"""
    
    @abstractmethod
    async def add_chunks(self, chunks: List[Chunk]) -> bool:
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
    async def process(self, file_path: str, file_type: str) -> List[Chunk]:
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
    async def create(self, filename: str, file_hash: str) -> Document:
        """Create new document record"""
        pass
    
    @abstractmethod
    async def get_by_id(self, document_id: str) -> Optional[Document]:
        """Get document by ID"""
        pass
    
    @abstractmethod
    async def get_by_hash(self, file_hash: str) -> Optional[Document]:
        """Check if document with hash exists"""
        pass
    
    @abstractmethod
    async def list_all(self) -> List[Document]:
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

# ============= Service Layer Interfaces =============
class IRAGService(ABC):
    """High-level RAG operations interface"""
    
    @abstractmethod
    async def process_document(self, file_path: str, filename: str, file_hash: str) -> Dict[str, Any]:
        """Process and store a document"""
        pass
    
    @abstractmethod
    async def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Search across all documents"""
        pass
    
    @abstractmethod
    async def delete_document(self, document_id: str) -> bool:
        """Delete a document and its chunks"""
        pass