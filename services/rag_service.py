# services/rag_service.py
import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile, Request
from api.schemas import ProcessDocumentResponse
from core.interfaces import (
    IDocumentProcessor, IRAGService, IVectorStore, IEmbeddingService, IDocumentRepository, 
    IMessageRepository, IFileStorage, DocumentChunk
)
from core.models import ChunkSearchResult, ProcessedDocument
from services.document_processor_factory import DocumentProcessorFactory
from utils.helpers import get_file_extension, get_file_hash, sanitize_filename, validate_file_content, validate_uploaded_file

logger = logging.getLogger(__name__)

class RAGService(IRAGService):
    def __init__(
        self,
        vector_store: IVectorStore,
        doc_processor_factory: DocumentProcessorFactory,
        embedding_service: IEmbeddingService,
        file_storage: IFileStorage,
        document_repo: IDocumentRepository,
        message_repo: IMessageRepository
    ):
        self.vector_store = vector_store
        self.doc_processor_factory = doc_processor_factory
        self.embedding_service = embedding_service
        self.file_storage = file_storage
        self.document_repo = document_repo
        self.message_repo = message_repo

    async def _prepare_file(self, file: UploadFile) -> Tuple[str, str, str]:
        '''
        Prepares an uploaded file for processing by validating, hashing, and generating a secure filename.
        
        This method performs pre-save validation and creates the necessary identifiers before
        the file is written to disk. It does NOT save the file - that happens in the caller.
        
        Args:
            file: The uploaded file object from FastAPI
            
        Returns:
            Tuple of (file_hash, document_id, stored_filename):
                - file_hash: SHA256 hash of file content for duplicate detection
                - document_id: UUID for database record
                - stored_filename: Secure filename for disk storage (UUID + sanitized extension)
                
        Raises:
            HTTPException: If file validation fails (missing filename, invalid type, too large)
        '''
        validate_uploaded_file(file)
        content = await file.read()
        await file.seek(0)
        
        doc_id = str(uuid4())
        
        safe_suffix = sanitize_filename(Path(file.filename).suffix)
        stored_name = f"{doc_id}{safe_suffix}"
        

        logger.info(f"✓ file is prepared (validation, hash, and secure filename)")
        return get_file_hash(content), doc_id, stored_name

    async def _process_chunks(self, file_path: str, file_type: str, document) -> List[DocumentChunk]:
        """
        Orchestrates text extraction, metadata enrichment, and embedding generation for a document.
        
        Coordinates the complete processing pipeline: extracts text chunks from the file using
        the appropriate OCR processor, enriches each chunk with document metadata, and generates
        vector embeddings for semantic search.
        
        Args:
            file_path: Absolute path to the document file on disk
            file_type: Normalized file extension ("pdf", "jpg", "jpeg", "png")
            document: ProcessedDocument instance containing document ID and metadata
            
        Returns:
            List[DocumentChunk]: Fully processed chunks ready for vector store insertion,
                                each containing text content, embeddings, document ID,
                                and metadata (document name, page number)
            
        Raises:
            ValueError: If no text content is extracted from the document
            Exception: If embedding generation fails
            
        Process Flow:
            1. Get appropriate processor for file type (PDF or image)
            2. Extract and split text into chunks via OCR
            3. Enrich each chunk with document_id and document_name metadata
            4. Generate embeddings for all chunks in batch
            5. Attach embeddings to corresponding chunks
            
        Example:
            chunks = await service._process_chunks(
                "/uploads/abc.pdf",
                "pdf", 
                document_record
            )
            
            Returns chunks ready for: await vector_store.add_chunks(chunks)
            
        Note:
            This method assumes the file has already been validated and saved to disk.
            Embedding generation happens in batch for efficiency rather than per-chunk.
            All chunks from a single document share the same document_id and document_name.
        """
        processor : IDocumentProcessor = self.doc_processor_factory.get_processor(file_type)
        chunks : List[DocumentChunk] = await processor.process(file_path, file_type)
        
        if not chunks:
            raise ValueError("No content extracted")

        # Set document info and generate embeddings
        for chunk in chunks:
            chunk.document_id = document.id
            chunk.metadata.update({"document_id": document.id, "document_name": document.filename})
        
        texts = [chunk.content for chunk in chunks]
        embeddings = await self.embedding_service.generate_embeddings(texts)
        
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
        
        return chunks

    async def _cleanup(self, document, stored_filename: str):
        """Clean up on failure - continue even if some operations fail"""
        if document and document.id:
            try: await self.vector_store.delete_by_document(document.id)
            except Exception as e: logger.warning(f"Vector cleanup failed: {e}")
            
            try: await self.document_repo.delete(document.id)
            except Exception as e: logger.warning(f"DB cleanup failed: {e}")
        
        if stored_filename:
            try: await self.file_storage.delete(stored_filename)
            except Exception as e: logger.warning(f"File cleanup failed: {e}")

    def _document_error_response(self, filename: str, error: str) -> ProcessDocumentResponse:
        """Helper for document processing errors only"""
        return ProcessDocumentResponse(
            status="error", 
            filename=filename, 
            document_id="",
            chunks=0, 
            pages=0, 
            error=error
        )

    async def process_document(self, file: UploadFile) -> ProcessDocumentResponse:
        try:
            file_hash, doc_id, stored_name = await self._prepare_file(file)
        except Exception as e:
            return self._document_error_response(file.filename, f"File validation failed: {str(e)}")
        
        if await self.document_repo.get_by_hash(file_hash):
            return self._document_error_response(file.filename, "Document already exists")

        document : Optional[ProcessedDocument] = None
        try:
            file_path = await self.file_storage.save(file, stored_name)
            
            validate_file_content(file_path, file.filename)
            
            document  = await self.document_repo.create(doc_id, file.filename, file_hash, stored_name)
            file_type : str = get_file_extension(file.filename)
            chunks : List[DocumentChunk] = await self._process_chunks(file_path, file_type, document)
            
            if not await self.vector_store.add_chunks(chunks):
                raise Exception("Failed to store vectors")
            
            return ProcessDocumentResponse(
                status="success", filename=file.filename, document_id=document.id,
                chunks=len(chunks), pages=len(set(c.metadata.get("page", 1) for c in chunks))
            )

        except Exception as e:
            logger.error(f"Processing failed for '{file.filename}': {e}")
            await self._cleanup(document, stored_name)
            return self._document_error_response(file.filename, str(e))

    async def search(self, query: str, top_k: int = 5) -> List[ChunkSearchResult]:
        try:
            query_embedding = await self.embedding_service.generate_query_embedding(query)
            results = await self.vector_store.search(query_embedding, top_k)
            
            # Filter valid results
            existing_docs = await self.document_repo.list_all()
            existing_ids = {doc.id for doc in existing_docs}
            valid_results = [
                result for result in results 
                if result.chunk.document_id in existing_ids
            ]
            
            # Save the search results using unified approach
            await self.message_repo.save_search_results(query, valid_results)
            
            return valid_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    async def delete_document(self, document_id: str) -> bool:
        try:
            document = await self.document_repo.get_by_id(document_id)
            if not document:
                return False
            
            # Delete in reverse order - vectors, file, then DB record
            stored_filename = document.metadata.get("stored_filename")
            
            success = True
            if stored_filename:
                try: await self.file_storage.delete(stored_filename)
                except Exception as e: 
                    logger.warning(f"File deletion failed: {e}")
                    success = False
            
            try: await self.vector_store.delete_by_document(document_id)
            except Exception as e: 
                logger.warning(f"Vector deletion failed: {e}")
                success = False
            
            try: await self.document_repo.delete(document_id)
            except Exception as e: 
                logger.error(f"DB deletion failed: {e}")
                return False  # This is critical
            
            return True  # DB record gone = success even if other cleanups failed
            
        except Exception as e:
            logger.error(f"Document deletion failed: {e}")
            return False
            
    async def list_documents(self, request: Request) -> List[Dict[str, str]]:
        documents = await self.document_repo.list_all()
        base_url = str(request.base_url)
        return [{
            "id": doc.id, 
            "filename": doc.filename,
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
        return {
            "original_filename": document.filename,
            "path": file_path
        } if file_path else None
    
    async def clear_all(self) -> bool:
        try:
            documents = await self.document_repo.list_all()
            
            # Delete physical files
            for doc in documents:
                stored_filename = doc.metadata.get("stored_filename")
                if stored_filename:
                    await self.file_storage.delete(stored_filename)
            
            # Clear vector store and database
            return (await self.vector_store.clear() and 
                    await self.document_repo.delete_all())
        except Exception as e:
            logger.error(f"Clear all failed: {e}")
            return False
    
    async def get_status(self) -> Dict[str, Any]:
        documents, chunk_count = await asyncio.gather(
            self.document_repo.list_all(),
            self.vector_store.count()
        )
        
        return {
            "document_loaded": ", ".join([d.filename for d in documents]) if documents else None,
            "chunks_available": chunk_count,
            "ready_for_queries": len(documents) > 0
        }