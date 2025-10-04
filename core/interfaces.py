# core/interfaces.py
"""Core interfaces for the RAG system"""
from abc import ABC, abstractmethod
from typing import Callable, List, Dict, Any, Optional
from dataclasses import dataclass
from fastapi import UploadFile, Request

from api.schemas import DocumentsListItem, ProcessDocumentResponse
from core.models import ChunkSearchResult, DocumentChunk, ProcessedDocument 

# ============= Vector Store Interface =============
class IVectorStore(ABC):
    """Interface for vector storage operations"""
    
    @abstractmethod
    async def add_chunks(self, chunks: List[DocumentChunk]) -> bool:
        """Add document chunks with embeddings"""
        pass
    
    @abstractmethod
    async def search(self, query_embedding: List[float], top_k: int = 5) -> List[ChunkSearchResult]:
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
    @abstractmethod
    async def process(
        self, 
        file_path: str, 
        file_type: str,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[DocumentChunk]:
        """Process document and return chunks"""
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
        """
        Creates a new document record in the database.
        
        Stores metadata about an uploaded document including its unique identifier,
        original filename, content hash for duplicate detection, and the secure filename
        used for disk storage.
        
        Args:
            document_id: Unique UUID identifier for the document
            filename: Original filename from the upload (user-visible name)
            file_hash: SHA256 hash of file content for duplicate detection
            stored_filename: Secure filename used on disk (UUID-based)
            
        Returns:
            ProcessedDocument: Domain model representing the created database record
            
        Example:
            doc = await repo.create(
                document_id="123e4567-e89b",
                filename="company_policy.pdf",
                file_hash="a3f2e1b...",
                stored_filename="123e4567-e89b.pdf"
            )
            
        Note:
            This only creates the database record. The actual file must be saved
            to disk separately using the file storage service. The stored_filename
            is needed later to locate and delete the physical file.
        """
        pass
    
    @abstractmethod
    async def get_by_id(self, document_id: str) -> Optional[ProcessedDocument]:
        """Get document by ID"""
        pass
    
    @abstractmethod
    async def get_by_hash(self, file_hash: str) -> Optional[ProcessedDocument]:
        """
        Retrieves a document by its content hash to detect duplicate uploads.
        
        Uses SHA256 hash comparison to find documents with identical content,
        even if filenames differ. This prevents processing the same document
        multiple times.
        
        Args:
            file_hash: SHA256 hash of the file content (64-character hex string)
            
        Returns:
            ProcessedDocument if a document with matching hash exists, None otherwise

        """
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
    async def save_search_results(self, query: str, results: List[ChunkSearchResult]) -> None:
        """Save search results in structured format"""
        pass

# ============= File Storage Interface =============
class IFileStorage(ABC):
    """Interface for physical file storage operations"""

    @abstractmethod
    async def save(self, file: UploadFile, filename: str) -> str:
        """
        Saves an uploaded file to the configured storage directory.
        
        Writes the file content to disk using the provided filename. Creates the
        upload directory if it doesn't exist. The filename should already be
        sanitized and unique (typically UUID-based) by the caller.
        
        Args:
            file: The uploaded file object containing content to save
            filename: Secure filename to use for storage (e.g., "uuid123.pdf")
            
        Returns:
            str: Full absolute path to the saved file
            
        Raises:
            Exception: If file cannot be written to disk (permissions, disk full, etc.)
            
        Example:
            file_path = await storage.save(uploaded_file, "abc-123.pdf")

            Returns: "/app/uploads/abc-123.pdf"
        """
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
    async def get_status(self) -> Dict[str, Any]:
        """Get system status"""
        pass
    
    @abstractmethod
    async def process_document(self, file: UploadFile) -> ProcessDocumentResponse:
        """
        Starts async document processing and returns immediately (< 1 second).
        
        Fast Path (synchronous):
            1. Validates file (size, type, format)
            2. Checks for duplicates (SHA256 hash)
            3. Saves file to disk
            4. Submits to background processor
            5. Returns with document_id for progress tracking
        
        Slow Path (background thread):
            - OCR text extraction (30s - 5min)
            - Embedding generation
            - Vector storage in ChromaDB
        
        Args:
            file: UploadFile (PDF, JPG, PNG, max 50MB)
        
        Returns:
            ProcessDocumentResponse with:
            - status: "processing" or "error"
            - document_id: UUID for polling /processing-status/{document_id}
            - error_code: FILE_TOO_LARGE, INVALID_FORMAT, DUPLICATE_FILE, etc.
        
        Error Handling:
            Validation failures return immediately with error_code.
            Processing failures tracked via ProgressStore.
            Auto-cleanup on any failure (file, DB, vectors).
        
        Concurrency:
            Handles 3 concurrent uploads via ThreadPoolExecutor.
        
        Example:
            response = await rag_service.process_document(file)
            # Returns: {"status": "processing", "document_id": "abc-123"}
            # Poll: GET /processing-status/abc-123
        """
        pass

    @abstractmethod
    async def search(self, query: str, top_k: int = 5) -> List[ChunkSearchResult]:
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
    async def list_documents(self, request: Request) -> List[DocumentsListItem]:
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
    def convert(self, file_path: str, dpi: int = 300) -> List[Any]:
        """Converts a PDF file to a list of images."""
        pass