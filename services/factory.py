# services/factory.py
"""Factory for creating service instances"""
from typing import Optional
import chromadb
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from core.interfaces import IRAGService
from infrastructure.vector_stores import ChromaDBVectorStore
from infrastructure.embedding_services import SentenceTransformerEmbedding
from infrastructure.repositories import SQLDocumentRepository, SQLMessageRepository
from services.rag_service_refactored import RAGService
from services.document_processor_factory import DocumentProcessorFactory # <-- NEW IMPORT

class ServiceFactory:
    """Factory for creating service instances with proper dependency injection"""
    
    _instances = {}
    
    @classmethod
    def create_rag_service(
        cls,
        session: AsyncSession,
        vector_store_type: str = "chromadb",
        embedding_model: Optional[str] = None
    ) -> IRAGService:
        """Create RAG service with specified implementations"""
        
        # Create or reuse vector store
        if "vector_store" not in cls._instances:
            if vector_store_type == "chromadb":
                client = chromadb.PersistentClient(path=settings.VECTOR_DB_PATH)
                cls._instances["vector_store"] = ChromaDBVectorStore(client)
            else:
                raise ValueError(f"Unknown vector store type: {vector_store_type}")
        
        # Create or reuse embedding service
        if "embedding_service" not in cls._instances:
            model_name = embedding_model or settings.EMBEDDING_MODEL_NAME
            cls._instances["embedding_service"] = SentenceTransformerEmbedding(model_name)
        
        # Create or reuse document processor factory
        if "doc_processor_factory" not in cls._instances:
            cls._instances["doc_processor_factory"] = DocumentProcessorFactory()
        
        # Create repositories with session
        document_repo = SQLDocumentRepository(session)
        message_repo = SQLMessageRepository(session)
        
        # Create and return service
        return RAGService(
            vector_store=cls._instances["vector_store"],
            doc_processor_factory=cls._instances["doc_processor_factory"], # <-- MODIFIED
            embedding_service=cls._instances["embedding_service"],
            document_repo=document_repo,
            message_repo=message_repo
        )
    
    @classmethod
    def clear_instances(cls):
        """Clear cached instances (useful for testing)"""
        cls._instances.clear()