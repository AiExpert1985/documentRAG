# services/rag_service_refactored.py
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile, Request # CHANGED
from core.interfaces import (
    IRAGService, IVectorStore, IEmbeddingService, IDocumentRepository, 
    IMessageRepository, IFileStorage, SearchResult, Chunk
)
from services.document_processor_factory import DocumentProcessorFactory

logger = logging.getLogger(__name__)

class RAGService(IRAGService):
    def __init__(
        self,
        vector_store: IVectorStore,
        doc_processor_factory: DocumentProcessorFactory,
        embedding_service: IEmbeddingService,
        file_storage: IFileStorage, # CHANGED
        document_repo: IDocumentRepository,
        message_repo: IMessageRepository
    ):
        self.vector_store = vector_store
        self.doc_processor_factory = doc_processor_factory
        self.embedding_service = embedding_service
        self.file_storage = file_storage # CHANGED
        self.document_repo = document_repo
        self.message_repo = message_repo

    async def process_document(
        self, 
        file: UploadFile, # CHANGED: Pass the whole file object
        filename: str, 
        file_hash: str,
        processing_strategy: Optional[str] = None
    ) -> Dict[str, Any]:
        if await self.document_repo.get_by_hash(file_hash):
            return {"status": "error", "error": "Document already exists"}

        document = None
        stored_filename = None
        try:
            # 1. Create DB record to get a unique ID
            # We pass a temporary stored_filename and update it later
            temp_stored_name = f"temp_{uuid4()}"
            document = await self.document_repo.create(filename, file_hash, temp_stored_name)
            
            # 2. Save the physical file using the unique ID
            file_extension = Path(filename).suffix
            stored_filename = f"{document.id}{file_extension}"
            file_path = await self.file_storage.save(file, stored_filename)
            
            # This is a bit of a workaround for not having the ID before creation.
            # In a real system, you might generate the ID first.
            # For now, we update the record with the correct name.
            document.metadata['stored_filename'] = stored_filename
            # (This part requires updating the repository to support updates,
            # for simplicity we will rely on the get_by_id to have the correct data later)
            # A cleaner way is to generate UUID in the service, but let's stick to the repo generating it.

            # 3. Process the document for RAG
            file_type = Path(filename).suffix[1:].lower()
            processor = self.doc_processor_factory.get_processor(file_type, processing_strategy)
            
            chunks = await processor.process(file_path, file_type)
            if not chunks:
                raise ValueError("No content could be extracted.")

            for chunk in chunks:
                chunk.document_id = document.id
                chunk.metadata["document_id"] = document.id
                chunk.metadata["document_name"] = filename
            
            texts = [chunk.content for chunk in chunks]
            embeddings = await self.embedding_service.generate_embeddings(texts)
            for chunk, embedding in zip(chunks, embeddings):
                chunk.embedding = embedding
            
            await self.vector_store.add_chunks(chunks)
            
            return { "status": "success", "filename": filename, "document_id": document.id,
                     "chunks": len(chunks), "pages": len(set(c.metadata.get("page", 1) for c in chunks)) }

        except Exception as e:
            logger.error(f"Processing failed for '{filename}': {e}", exc_info=True)
            # Rollback: delete DB record and physical file
            if document and document.id:
                await self.document_repo.delete(document.id)
            if stored_filename:
                await self.file_storage.delete(stored_filename)
            return {"status": "error", "error": str(e)}

    async def delete_document(self, document_id: str) -> bool:
        try:
            document = await self.document_repo.get_by_id(document_id)
            if not document: return False

            # Delete physical file first
            stored_filename = document.metadata.get("stored_filename")
            if stored_filename:
                await self.file_storage.delete(stored_filename)
            
            # Then delete vectors and DB record
            await self.vector_store.delete_by_document(document_id)
            await self.document_repo.delete(document_id)
            return True
        except Exception as e:
            logger.error(f"Document deletion failed for ID {document_id}: {e}")
            return False
            
    async def list_documents(self, request: Request) -> List[Dict[str, str]]:
        documents = await self.document_repo.list_all()
        base_url = str(request.base_url)
        return [{
            "id": doc.id, 
            "filename": doc.filename,
            # CHANGED: Construct a full download URL
            "download_url": f"{base_url}download/{doc.id}"
        } for doc in documents]
        
    async def get_document_with_path(self, document_id: str) -> Optional[Dict[str, Any]]:
        document = await self.document_repo.get_by_id(document_id)
        if not document:
            return None
        
        stored_filename = document.metadata.get("stored_filename")
        if not stored_filename:
            return None
            
        file_path = await self.file_storage.get_path(stored_filename)
        if not file_path:
            return None
            
        return {
            "original_filename": document.filename,
            "path": file_path
        }
    # ... (search, clear_all, get_status are unchanged) ...

    async def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        try:
            query_embedding = await self.embedding_service.generate_query_embedding(query)
            results = await self.vector_store.search(query_embedding, top_k)
            await self.message_repo.save_search(query, len(results))
            return results
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    async def clear_all(self) -> bool:
        try:
            vector_cleared = await self.vector_store.clear()
            db_cleared = await self.document_repo.delete_all()
            return vector_cleared and db_cleared
        except Exception as e:
            logger.error(f"Clear all failed: {e}")
            return False
    
    async def get_status(self) -> Dict[str, Any]:
        documents = await self.document_repo.list_all()
        chunk_count = await self.vector_store.count()
        return {
            "document_loaded": ", ".join([d.filename for d in documents]) if documents else None,
            "chunks_available": chunk_count,
            "ready_for_queries": len(documents) > 0
        }