# services/rag_service.py
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile, Request, HTTPException
from pydantic import HttpUrl
from api.schemas import DocumentsListItem, ProcessDocumentResponse, ErrorCode, ProcessingStatus
from core.interfaces import (
    IDocumentProcessor, IRAGService, IVectorStore, IEmbeddingService, 
    IDocumentRepository, IMessageRepository, IFileStorage, DocumentChunk
)
from core.models import ChunkSearchResult, ProcessedDocument
from services.document_processor_factory import DocumentProcessorFactory
from utils.helpers import (
    get_file_extension, get_file_hash, sanitize_filename, 
    validate_file_content, validate_uploaded_file
)
from infrastructure.progress_store import progress_store
from services.async_processor import async_processor

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

    # ============ VALIDATION & PREPARATION ============
    
    async def _validate_and_prepare(self, file: UploadFile) -> Tuple[str, str, str, bytes]:
        """Validate file and generate IDs. Returns (hash, doc_id, stored_name, content)"""
        if not file.filename:
            raise ValueError(f"{ErrorCode.INVALID_FORMAT.value}:No filename provided")
        try:
            validate_uploaded_file(file)
        except HTTPException as e:
            error_code = self._map_http_error_to_code(e.status_code)
            raise ValueError(f"{error_code.value}:{e.detail}")
        
        content = await file.read()
        await file.seek(0)
        
        doc_id = str(uuid4())
        safe_suffix = sanitize_filename(Path(file.filename).suffix)
        stored_name = f"{doc_id}{safe_suffix}"
        
        return get_file_hash(content), doc_id, stored_name, content
    
    def _map_http_error_to_code(self, status_code: int) -> ErrorCode:
        """Map HTTP status to error code"""
        if status_code == 413:
            return ErrorCode.FILE_TOO_LARGE
        elif status_code == 400:
            return ErrorCode.INVALID_FORMAT
        return ErrorCode.PROCESSING_FAILED
    
    # ============ FILE OPERATIONS ============
    
    async def _check_duplicate(self, file_hash: str, filename: str) -> None:
        """Check if file already exists"""
        existing = await self.document_repo.get_by_hash(file_hash)
        if existing:
            raise ValueError(f"{ErrorCode.DUPLICATE_FILE.value}:Document already exists")
    
    async def _save_and_validate_file(self, file: UploadFile, stored_name: str) -> str:
        """Save file to disk and validate content"""
        file_path = await self.file_storage.save(file, stored_name)
        
        assert file.filename is not None, "Filename should not be None at this point"
        
        try:
            validate_file_content(file_path, file.filename)
        except HTTPException as e:
            await self.file_storage.delete(stored_name)
            raise ValueError(f"{ErrorCode.INVALID_FORMAT.value}:{e.detail}")
        
        return file_path
    
    # ============ DOCUMENT PROCESSING ============
    
    async def _extract_and_embed_chunks(
        self, file_path: str, file_type: str, document: ProcessedDocument, doc_id: str
    ) -> List[DocumentChunk]:
        """Extract text and generate embeddings"""
        progress_store.update(doc_id, ProcessingStatus.EXTRACTING_TEXT, 30, 
                            "Extracting text from document...")
        
        processor: IDocumentProcessor = self.doc_processor_factory.get_processor(file_type)
        
        try:
            chunks: List[DocumentChunk] = await processor.process(file_path, file_type)
        except TimeoutError:
            raise ValueError(f"{ErrorCode.OCR_TIMEOUT.value}:Text extraction timed out")
        except Exception as e:
            raise ValueError(f"{ErrorCode.NO_TEXT_FOUND.value}:No text extracted - {str(e)}")
        
        if not chunks:
            raise ValueError(f"{ErrorCode.NO_TEXT_FOUND.value}:No content extracted from document")
        
        # Enrich chunks with metadata
        for chunk in chunks:
            chunk.document_id = document.id
            chunk.metadata.update({
                "document_id": document.id, 
                "document_name": document.filename
            })
        
        # Generate embeddings
        progress_store.update(doc_id, ProcessingStatus.GENERATING_EMBEDDINGS, 60, 
                            f"Generating embeddings for {len(chunks)} chunks...")
        
        texts = [chunk.content for chunk in chunks]
        embeddings = await self.embedding_service.generate_embeddings(texts)
        
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
        
        return chunks
    
    # ============ STORAGE ============
    
    async def _store_chunks(self, chunks: List[DocumentChunk], doc_id: str) -> None:
        """Store chunks in vector database"""
        progress_store.update(doc_id, ProcessingStatus.STORING, 80, 
                            "Storing in vector database...")
        
        success = await self.vector_store.add_chunks(chunks)
        if not success:
            raise ValueError(f"{ErrorCode.PROCESSING_FAILED.value}:Failed to store vectors")
    
    # ============ CLEANUP ============
    
    async def _cleanup_on_failure(self, document: Optional[ProcessedDocument], 
                                  stored_filename: str) -> None:
        """Clean up resources after failure"""
        if document and document.id:
            try:
                await self.vector_store.delete_by_document(document.id)
            except Exception as e:
                logger.warning(f"Vector cleanup failed: {e}")
            
            try:
                await self.document_repo.delete(document.id)
            except Exception as e:
                logger.warning(f"DB cleanup failed: {e}")
        
        if stored_filename:
            try:
                await self.file_storage.delete(stored_filename)
            except Exception as e:
                logger.warning(f"File cleanup failed: {e}")
    
    # ============ ERROR RESPONSE BUILDER ============
    
    def _build_error_response(self, filename: Optional[str], error_msg: str) -> ProcessDocumentResponse:
        """Build error response with proper error code"""
        if ":" in error_msg:
            code_str, message = error_msg.split(":", 1)
            try:
                error_code = ErrorCode(code_str)
            except ValueError:
                error_code = ErrorCode.PROCESSING_FAILED
                message = error_msg
        else:
            error_code = ErrorCode.PROCESSING_FAILED
            message = error_msg
        
        return ProcessDocumentResponse(
            status="error",
            filename=filename or "unknown",  # Handle None case
            document_id="",
            chunks=0,
            pages=0,
            error=message,
            error_code=error_code
        )
        
        return ProcessDocumentResponse(
            status="error",
            filename=filename,
            document_id="",
            chunks=0,
            pages=0,
            error=message,
            error_code=error_code
        )
    
    # ============ BACKGROUND PROCESSING ============
    
    async def _process_document_background(
        self, doc_id: str, file_path: str, file_type: str, 
        file_hash: str, filename: str, stored_name: str
    ) -> None:
        """The actual heavy processing - runs in background thread"""
        document: Optional[ProcessedDocument] = None
        
        try:
            # Create DB record
            document = await self.document_repo.create(
                doc_id, filename, file_hash, stored_name
            )
            
            # Extract and embed (heavy operations)
            chunks = await self._extract_and_embed_chunks(file_path, file_type, document, doc_id)
            
            # Store in vector DB
            await self._store_chunks(chunks, doc_id)
            
            # Success!
            progress_store.complete(doc_id)
            logger.info(f"✅ Successfully processed {filename}")
        
        except ValueError as e:
            error_msg = str(e)
            logger.error(f"Processing failed for '{filename}': {error_msg}")
            
            code_str = error_msg.split(":")[0] if ":" in error_msg else "PROCESSING_FAILED"
            try:
                error_code = ErrorCode(code_str)
            except ValueError:
                error_code = ErrorCode.PROCESSING_FAILED
            
            progress_store.fail(doc_id, error_msg.split(":", 1)[-1], error_code)
            await self._cleanup_on_failure(document, stored_name)
        
        except Exception as e:
            logger.exception(f"Unexpected error processing '{filename}'")
            progress_store.fail(doc_id, str(e), ErrorCode.PROCESSING_FAILED)
            await self._cleanup_on_failure(document, stored_name)
    
    # ============ MAIN PROCESSING METHOD (ASYNC - NON-BLOCKING) ============
    
    async def process_document(self, file: UploadFile) -> ProcessDocumentResponse:
        """
        Start document processing in background.
        Returns immediately with document_id for progress tracking.
        """
        try:
            # Step 1: Fast validation and file save
            file_hash, doc_id, stored_name, _ = await self._validate_and_prepare(file)
            
            # Assert filename is not None (validated in _validate_and_prepare)
            assert file.filename is not None
            
            progress_store.start(doc_id, file.filename)
            progress_store.update(doc_id, ProcessingStatus.VALIDATING, 10, "Validating file...")
            
            # Step 2: Check duplicate
            await self._check_duplicate(file_hash, file.filename)
            
            # Step 3: Save file to disk
            progress_store.update(doc_id, ProcessingStatus.VALIDATING, 20, "Saving file...")
            file_path = await self._save_and_validate_file(file, stored_name)
            file_type = get_file_extension(file.filename)
            
            # Step 4: Submit to background processing (RETURNS IMMEDIATELY)
            async_processor.submit_task(
                self._process_document_background(
                    doc_id, file_path, file_type, file_hash, file.filename, stored_name
                )
            )
            
            # Return immediately with document_id
            return ProcessDocumentResponse(
                status="processing",
                filename=file.filename,
                document_id=doc_id,
                chunks=0,
                pages=0,
                message="Document is being processed in background"
            )
        
        except ValueError as e:
            error_msg = str(e)
            logger.error(f"Upload validation failed for '{file.filename}': {error_msg}")
            return self._build_error_response(file.filename, error_msg)
        
        except Exception as e:
            error_msg = f"{ErrorCode.PROCESSING_FAILED.value}:Unexpected error - {str(e)}"
            logger.exception(f"Unexpected error uploading '{file.filename}'")
            return self._build_error_response(file.filename, error_msg)

    # ============ OTHER METHODS (UNCHANGED) ============

    async def search(self, query: str, top_k: int = 5) -> List[ChunkSearchResult]:
        """Search across all documents"""
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
            
            # Save search results
            await self.message_repo.save_search_results(query, valid_results)
            return valid_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    async def delete_document(self, document_id: str) -> bool:
        """Delete a document and its chunks"""
        try:
            document = await self.document_repo.get_by_id(document_id)
            if not document:
                return False
            
            stored_filename = document.metadata.get("stored_filename")
            
            success = True
            if stored_filename:
                try:
                    await self.file_storage.delete(stored_filename)
                except Exception as e:
                    logger.warning(f"File deletion failed: {e}")
                    success = False
            
            try:
                await self.vector_store.delete_by_document(document_id)
            except Exception as e:
                logger.warning(f"Vector deletion failed: {e}")
                success = False
            
            try:
                await self.document_repo.delete(document_id)
            except Exception as e:
                logger.error(f"DB deletion failed: {e}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Document deletion failed: {e}")
            return False
            
    async def list_documents(self, request: Request) -> List[DocumentsListItem]:
        """List all documents, including download URLs"""
        documents = await self.document_repo.list_all()
        base_url = str(request.base_url)
        return [
            DocumentsListItem(
                id=doc.id,
                filename=doc.filename,
                download_url=HttpUrl(f"{base_url}download/{doc.id}")
            )
            for doc in documents
        ]
        
    async def get_document_with_path(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get a document's details and its physical file path"""
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
        """Clear all documents and vectors"""
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
        """Get system status"""
        import asyncio
        documents, chunk_count = await asyncio.gather(
            self.document_repo.list_all(),
            self.vector_store.count()
        )
        
        return {
            "document_loaded": ", ".join([d.filename for d in documents]) if documents else None,
            "chunks_available": chunk_count,
            "ready_for_queries": len(documents) > 0
        }