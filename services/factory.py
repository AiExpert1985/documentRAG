# services/factory.py
from typing import Optional
import chromadb
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from core.interfaces import IRAGService
from infrastructure.vector_stores import ChromaDBVectorStore
from infrastructure.embedding_services import SentenceTransformerEmbedding
from infrastructure.repositories import SQLDocumentRepository, SQLMessageRepository
from infrastructure.file_storage import LocalFileStorage
from services.rag_service_refactored import RAGService
from services.document_processor_factory import DocumentProcessorFactory

class ServiceFactory:
    _instances = {}
    
    @classmethod
    def create_rag_service(
        cls,
        session: AsyncSession,
        vector_store_type: str = "chromadb",
        embedding_model: Optional[str] = None
    ) -> IRAGService:
        if "vector_store" not in cls._instances:
            if vector_store_type == "chromadb":
                client = chromadb.PersistentClient(path=settings.VECTOR_DB_PATH)
                cls._instances["vector_store"] = ChromaDBVectorStore(client)
            else:
                raise ValueError(f"Unknown vector store type: {vector_store_type}")
        
        if "embedding_service" not in cls._instances:
            model_name = embedding_model or settings.EMBEDDING_MODEL_NAME
            cls._instances["embedding_service"] = SentenceTransformerEmbedding(model_name)

        if "file_storage" not in cls._instances:
            cls._instances["file_storage"] = LocalFileStorage(base_path=settings.UPLOADS_DIR)
        
        if "doc_processor_factory" not in cls._instances:
            cls._instances["doc_processor_factory"] = DocumentProcessorFactory()
        
        document_repo = SQLDocumentRepository(session)
        message_repo = SQLMessageRepository(session)
        
        return RAGService(
            vector_store=cls._instances["vector_store"],
            doc_processor_factory=cls._instances["doc_processor_factory"],
            embedding_service=cls._instances["embedding_service"],
            file_storage=cls._instances["file_storage"], # <<< --- THIS LINE WAS MISSING
            document_repo=document_repo,
            message_repo=message_repo
        )
    
    @classmethod
    def clear_instances(cls):
        cls._instances.clear()