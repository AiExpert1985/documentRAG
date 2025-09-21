# services/factory.py
from typing import Optional
import chromadb
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import Depends

from api.endpoints_refactored import get_db
from config import settings
from core.interfaces import (
    IRAGService, IVectorStore, IEmbeddingService, 
    IDocumentRepository, IMessageRepository, IFileStorage
)
from infrastructure.vector_stores import ChromaDBVectorStore
from infrastructure.embedding_services import SentenceTransformerEmbedding
from infrastructure.repositories import SQLDocumentRepository, SQLMessageRepository
from infrastructure.file_storage import LocalFileStorage
from services.rag_service_refactored import RAGService
from services.document_processor_factory import DocumentProcessorFactory

# Provider functions for each component
def get_vector_store() -> IVectorStore:
    """Create vector store based on configuration."""
    if settings.VECTOR_STORE_TYPE == "chromadb":
        client = chromadb.PersistentClient(path=settings.VECTOR_DB_PATH)
        return ChromaDBVectorStore(client)
    # Future: elif settings.VECTOR_STORE_TYPE == "pinecone":
    #     return PineconeVectorStore(...)
    else:
        raise ValueError(f"Unknown vector store type: {settings.VECTOR_STORE_TYPE}")

def get_embedding_service() -> IEmbeddingService:
    """Create embedding service based on configuration."""
    return SentenceTransformerEmbedding(settings.EMBEDDING_MODEL_NAME)
    # Future: if settings.EMBEDDING_PROVIDER == "openai":
    #     return OpenAIEmbeddingService(...)

def get_file_storage() -> IFileStorage:
    """Create file storage based on configuration."""
    return LocalFileStorage(base_path=settings.UPLOADS_DIR)
    # Future: if settings.STORAGE_TYPE == "s3":
    #     return S3FileStorage(...)

def get_document_processor_factory() -> DocumentProcessorFactory:
    """Create document processor factory."""
    return DocumentProcessorFactory()

def get_document_repository(session: AsyncSession = Depends(get_db)) -> IDocumentRepository:
    """Create document repository with injected session."""
    return SQLDocumentRepository(session)

def get_message_repository(session: AsyncSession = Depends(get_db)) -> IMessageRepository:
    """Create message repository with injected session."""
    return SQLMessageRepository(session)

# Main service provider using FastAPI DI
def get_rag_service(
    session: AsyncSession = Depends(get_db),
    vector_store: IVectorStore = Depends(get_vector_store),
    embedding_service: IEmbeddingService = Depends(get_embedding_service),
    file_storage: IFileStorage = Depends(get_file_storage),
    doc_processor_factory: DocumentProcessorFactory = Depends(get_document_processor_factory),
    document_repo: IDocumentRepository = Depends(get_document_repository),
    message_repo: IMessageRepository = Depends(get_message_repository)
) -> IRAGService:
    """
    Create RAG service with full dependency injection.
    
    FastAPI automatically provides all dependencies based on their providers.
    Easy to override individual components for testing.
    """
    return RAGService(
        vector_store=vector_store,
        doc_processor_factory=doc_processor_factory,
        embedding_service=embedding_service,
        file_storage=file_storage,
        document_repo=document_repo,
        message_repo=message_repo
    )

# Remove the ServiceFactory class - replaced by get_rag_service function