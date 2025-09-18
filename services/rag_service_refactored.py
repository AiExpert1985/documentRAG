# services/rag_service_refactored.py
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from core.interfaces import (
    IRAGService, IVectorStore, 
    IEmbeddingService, IDocumentRepository, IMessageRepository,
    SearchResult, Chunk
)
from services.document_processor_factory import DocumentProcessorFactory

logger = logging.getLogger(__name__)

class RAGService(IRAGService):
    """Main RAG service orchestrating all operations"""
    
    def __init__(
        self,
        vector_store: IVectorStore,
        doc_processor_factory: DocumentProcessorFactory,
        embedding_service: IEmbeddingService,
        document_repo: IDocumentRepository,
        message_repo: IMessageRepository
    ):
        self.vector_store = vector_store
        self.doc_processor_factory = doc_processor_factory
        self.embedding_service = embedding_service
        self.document_repo = document_repo
        self.message_repo = message_repo

    # CHANGED: Broke down the original process_document into smaller, private methods
    async def _validate_and_get_processor(self, file_path: str, file_type: str, strategy: Optional[str]):
        """Validates the document and returns the correct processor."""
        final_strategy = strategy
        if not final_strategy or final_strategy == "auto":
            if file_type == "pdf":
                final_strategy = self.doc_processor_factory.detect_pdf_strategy(file_path)
        
        processor = self.doc_processor_factory.get_processor(file_type, final_strategy)
        
        if not await processor.validate(file_path, file_type):
            raise ValueError(f"Invalid or corrupted {file_type} file")
            
        return processor

    async def _create_chunks(self, processor, file_path: str, file_type: str, document, filename: str) -> List[Chunk]:
        """Creates, enriches, and returns document chunks."""
        chunks = await processor.process(file_path, file_type)
        if not chunks:
            raise ValueError("No content could be extracted from the document.")

        for chunk in chunks:
            chunk.document_id = document.id
            chunk.metadata["document_id"] = document.id
            chunk.metadata["document_name"] = filename
        return chunks

    async def _embed_and_store_chunks(self, chunks: List[Chunk]):
        """Generates embeddings and stores chunks in the vector store."""
        texts = [chunk.content for chunk in chunks]
        embeddings = await self.embedding_service.generate_embeddings(texts)
        
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
            
        if not await self.vector_store.add_chunks(chunks):
            raise RuntimeError("Failed to store chunks in vector database")

    async def process_document(
        self, 
        file_path: str, 
        filename: str, 
        file_hash: str,
        processing_strategy: Optional[str] = None
    ) -> Dict[str, Any]:
        """Orchestrates the document processing pipeline."""
        if await self.document_repo.get_by_hash(file_hash):
            return {"status": "error", "error": "Document already exists"}

        document = None
        try:
            file_type = Path(filename).suffix[1:].lower()
            
            processor = await self._validate_and_get_processor(file_path, file_type, processing_strategy)
            
            # Create DB entry first to reserve the spot
            document = await self.document_repo.create(filename, file_hash)
            
            # Process and store chunks
            chunks = await self._create_chunks(processor, file_path, file_type, document, filename)
            await self._embed_and_store_chunks(chunks)
            
            return {
                "status": "success", "filename": filename, "document_id": document.id,
                "chunks": len(chunks), "pages": len(set(c.metadata.get("page", 1) for c in chunks))
            }
        except Exception as e:
            logger.error(f"Processing failed for '{filename}': {e}", exc_info=True)
            # Rollback DB entry if it was created and processing failed
            if document and document.id:
                await self.document_repo.delete(document.id)
            return {"status": "error", "error": str(e)}

    async def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        try:
            query_embedding = await self.embedding_service.generate_query_embedding(query)
            results = await self.vector_store.search(query_embedding, top_k)
            await self.message_repo.save_search(query, len(results))
            return results
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    async def delete_document(self, document_id: str) -> bool:
        try:
            document = await self.document_repo.get_by_id(document_id)
            if not document: return False
            vector_deleted = await self.vector_store.delete_by_document(document_id)
            db_deleted = await self.document_repo.delete(document_id)
            return vector_deleted and db_deleted
        except Exception as e:
            logger.error(f"Document deletion failed: {e}")
            return False
    
    async def clear_all(self) -> bool:
        try:
            vector_cleared = await self.vector_store.clear()
            db_cleared = await self.document_repo.delete_all()
            return vector_cleared and db_cleared
        except Exception as e:
            logger.error(f"Clear all failed: {e}")
            return False
    
    async def list_documents(self) -> List[Dict[str, str]]:
        documents = await self.document_repo.list_all()
        return [{"id": doc.id, "filename": doc.filename} for doc in documents]
    
    async def get_status(self) -> Dict[str, Any]:
        documents = await self.document_repo.list_all()
        chunk_count = await self.vector_store.count()
        return {
            "document_loaded": ", ".join([d.filename for d in documents]) if documents else None,
            "chunks_available": chunk_count,
            "ready_for_queries": len(documents) > 0
        }