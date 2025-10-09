# services/rag_service.py
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile, Request, HTTPException
from pydantic import HttpUrl

from api.schemas import (
    ProcessDocumentResponse, ErrorCode, ProcessingStatus, 
    DocumentProcessingError, DocumentsListItem
)
from core.domain import DocumentResponseStatus
from core.interfaces import (
    IDocumentProcessor, IRAGService, IVectorStore, IEmbeddingService, 
    IDocumentRepository, IMessageRepository, IFileStorage, DocumentChunk
)
from core.domain import ChunkSearchResult, ProcessedDocument
from services.document_processor_factory import DocumentProcessorFactory
from database.session import get_session  # NEW
from infrastructure.repositories import SQLDocumentRepository  # NEW
from utils.common import (
    get_file_extension, get_file_hash, sanitize_filename, 
    validate_file_content, validate_uploaded_file
)
from infrastructure.progress_store import progress_store
from services.async_processor import async_processor

from config import settings

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
        self.document_repo = document_repo
        self.message_repo = message_repo
        self.vector_store = vector_store
        self.doc_processor_factory = doc_processor_factory
        self.embedding_service = embedding_service
        self.file_storage = file_storage

    # ============ VALIDATION & PREPARATION ============
    
    async def _validate_and_prepare(self, file: UploadFile) -> Tuple[str, str, str, bytes]:
        """
        Validate file (size, type, format) and generate IDs. No DB queries.
        Returns (file_hash, doc_id, stored_filename, content).
        """
        if not file.filename:
            raise DocumentProcessingError(
                "No filename provided", 
                ErrorCode.INVALID_FORMAT
            )

        try:
            validate_uploaded_file(file)
        except HTTPException as e:
            error_code = (
                ErrorCode.FILE_TOO_LARGE if e.status_code == 413 else
                ErrorCode.INVALID_FORMAT if e.status_code == 400 else
                ErrorCode.PROCESSING_FAILED
            )
            raise DocumentProcessingError(e.detail, error_code)
        
        content = await file.read()
        await file.seek(0)
        
        doc_id = str(uuid4())
        safe_suffix = sanitize_filename(Path(file.filename).suffix)
        stored_name = f"{doc_id}{safe_suffix}"
        
        return get_file_hash(content), doc_id, stored_name, content
    
    # ============ FILE OPERATIONS ============
    
    async def _check_duplicate(self, file_hash: str, filename: str) -> None:
        """Check if file already exists"""
        existing = await self.document_repo.get_by_hash(file_hash)
        if existing:
            raise DocumentProcessingError(
                "Document already exists",
                ErrorCode.DUPLICATE_FILE
            )
    
    async def _save_and_validate_file(self, file: UploadFile, stored_name: str) -> str:
        """
        Save file to disk and validate content (magic numbers). 
        Auto-cleanup on failure. Returns absolute file path.
        """
        file_path = await self.file_storage.save(file, stored_name)
        assert file.filename is not None, "Filename should not be None at this point"   
        try:
            validate_file_content(file_path, file.filename)
        except HTTPException as e:
            await self.file_storage.delete(stored_name)
            raise DocumentProcessingError(e.detail, ErrorCode.INVALID_FORMAT)
        
        return file_path
    
    # ============ DOCUMENT PROCESSING ============
    async def _extract_text_chunks(
        self, file_path: str, file_type: str, document: ProcessedDocument, doc_id: str
    ) -> List[DocumentChunk]:
        """
        Extract text via OCR and split into chunks (without embeddings).
        Progress: 30-95%. Returns chunks with embedding=None.
        """
        def update_page_progress(current_page: int, total_pages: int):
            page_percent = (current_page / total_pages) * 65
            overall_percent = 30 + int(page_percent)
            progress_store.update(
                doc_id, 
                ProcessingStatus.EXTRACTING_TEXT, 
                overall_percent,
                f"Extracting text from page {current_page}/{total_pages}..."
            )
        
        progress_store.update(doc_id, ProcessingStatus.EXTRACTING_TEXT, 30, 
                            "Starting text extraction...")
        
        processor = self.doc_processor_factory.get_processor(file_type)
        chunks = await processor.process(file_path, file_type, update_page_progress)
        
        if not chunks:
            raise DocumentProcessingError(
                "No content extracted from document", 
                ErrorCode.NO_TEXT_FOUND
            )
        
        for chunk in chunks:
            chunk.document_id = document.id
            chunk.metadata.update({
                "document_id": document.id, 
                "document_name": document.filename
            })
        
        # 🔍 DEBUG: Save chunks to file for inspection
        self._save_chunks_for_debug(chunks, document.filename)
        
        return chunks


    async def _generate_embeddings(
        self, chunks: List[DocumentChunk], doc_id: str
    ) -> List[DocumentChunk]:
        """
        Generate and attach embeddings to chunks.
        
        Takes chunks without embeddings, returns same chunks WITH embeddings.
        Progress: 95%
        """
        progress_store.update(
            doc_id, 
            ProcessingStatus.GENERATING_EMBEDDINGS, 
            95, 
            f"Generating embeddings for {len(chunks)} chunks..."
        )
        
        texts : List[str] = [chunk.content for chunk in chunks]
        embeddings: List[List[float]] = await self.embedding_service.generate_embeddings(texts)
        
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
        
        return chunks

    
    # ============ STORAGE ============
    
    async def _store_chunks(self, chunks: List[DocumentChunk], doc_id: str) -> None:
        """
        Store chunks in vector database. Progress: 80%.
        Raises DocumentProcessingError if storage fails.
        """
        progress_store.update(doc_id, ProcessingStatus.STORING, 80, 
                            "Storing in vector database...")
        
        success = await self.vector_store.add_chunks(chunks)
        if not success:
            raise DocumentProcessingError(
                "Failed to store vectors", 
                ErrorCode.PROCESSING_FAILED
            )
    
    # ============ CLEANUP ============
    
    async def _cleanup_on_failure(
        self, 
        document_id: Optional[str],
        stored_filename: str, 
        doc_id: str,
        doc_repo: Optional[IDocumentRepository] = None  # MODIFIED
    ) -> None:
        """
        Rollback all changes after processing failure (file, database, vectors).
        
        Attempts to delete in reverse order of creation. Continues cleanup even if
        individual steps fail (logs warnings, doesn't raise exceptions).
        
        Args:
            document_id: Database record ID (None if DB record not created yet)
            stored_filename: Physical filename on disk (e.g., "uuid.pdf")
            doc_id: Progress tracking ID (always exists)
            doc_repo: Database repository (uses independent session in background)
            
        Cleanup Order:
            1. Vector store chunks (if document_id exists)
            2. Database record (if document_id exists)
            3. Physical file (if stored_filename exists)
            4. Progress store entry (always)
            
        Error Handling:
            - Logs warnings if cleanup steps fail
            - Never raises exceptions (best-effort cleanup)
            - Continues attempting all steps even if one fails
            
        Example:
            # After OCR timeout, rollback everything:
            await self._cleanup_on_failure(
                document.id,      # DB record exists
                "uuid.pdf",       # File exists
                doc_id,           # Progress exists
                doc_repo          # Background session
            )
        """
        repo = doc_repo or self.document_repo  
        
        if document_id:
            try:
                await self.vector_store.delete_by_document(document_id)
            except Exception as e:
                logger.warning(f"Vector cleanup failed: {e}")
            
            try:
                await repo.delete(document_id)  
            except Exception as e:
                logger.warning(f"DB cleanup failed: {e}")
        
        if stored_filename:
            try:
                await self.file_storage.delete(stored_filename)
            except Exception as e:
                logger.warning(f"File cleanup failed: {e}")
        
        progress_store.remove(doc_id)
        logger.info(f"Cleanup complete for {doc_id}")
    
    # ============ BACKGROUND PROCESSING ============
    
    async def _process_document_background(
        self, doc_id: str, file_path: str, file_type: str, 
        file_hash: str, filename: str, stored_name: str
    ) -> None:
        """
        Complete document processing pipeline in background (independent DB session).
        
        Pipeline: duplicate check → OCR → embeddings → storage.
        Updates progress_store throughout. Auto-cleanup on failure. Never raises (logs errors).
        """
        document: Optional[ProcessedDocument] = None
        
        async with get_session() as session:
            doc_repo = SQLDocumentRepository(session)
            
            try:
                # Check duplicate
                existing = await doc_repo.get_by_hash(file_hash)
                if existing:
                    raise DocumentProcessingError(
                        "Document already exists",
                        ErrorCode.DUPLICATE_FILE
                    )
                
                # Create document
                document = await doc_repo.create(
                    doc_id, filename, file_hash, stored_name
                )
                
                # 🔹 Step 1: Extract text (30-95%)
                chunks : List[DocumentChunk] = await self._extract_text_chunks(
                    file_path, file_type, document, doc_id
                )

                for i, chunk in enumerate(chunks):
                    print(f"{i}: {chunk.content}")
                
                # 🔹 Step 2: Generate embeddings (95%)
                chunks = await self._generate_embeddings(chunks, doc_id)
                
                # 🔹 Step 3: Store in vector DB
                await self._store_chunks(chunks, doc_id)
                
                progress_store.complete(doc_id)
                logger.info(f"Successfully processed {filename}")
            
            except DocumentProcessingError as e:
                logger.error(f"Processing failed for '{filename}': {e.message}")
                progress_store.fail(doc_id, e.message, e.error_code)
                await self._cleanup_on_failure(
                    document.id if document else None,
                    stored_name, 
                    doc_id,
                    doc_repo
                )
            
            except Exception as e:
                logger.exception(f"Unexpected error processing '{filename}'")
                progress_store.fail(
                    doc_id, 
                    f"System error: {str(e)[:100]}",
                    ErrorCode.PROCESSING_FAILED
                )
                await self._cleanup_on_failure(
                    document.id if document else None,
                    stored_name, 
                    doc_id,
                    doc_repo
                )
        


    
    # ============ MAIN PROCESSING METHOD ============
    
    async def process_document(self, file: UploadFile) -> ProcessDocumentResponse:
        doc_id: Optional[str] = None
        
        try:
            file_hash, doc_id, stored_name, _ = await self._validate_and_prepare(file)
            assert file.filename is not None
            progress_store.start(doc_id, file.filename)
            
            progress_store.update(
                doc_id, ProcessingStatus.VALIDATING, 20, "Saving file..."
            )
            file_path = await self._save_and_validate_file(file, stored_name)
            file_type = get_file_extension(file.filename)
            
            async_processor.submit_task(
                self._process_document_background(
                    doc_id, file_path, file_type, file_hash, 
                    file.filename, stored_name
                )
            )
            
            return ProcessDocumentResponse(
                status=DocumentResponseStatus.PROCESSING,
                filename=file.filename,
                document_id=doc_id,
                chunks=0,
                pages=0,
                message="Document is being processed in background"
            )
        
        except DocumentProcessingError as e:
            logger.error(f"Upload validation failed: {e.message}")
            if doc_id:
                progress_store.fail(doc_id, e.message, e.error_code)
            return ProcessDocumentResponse(
                status=DocumentResponseStatus.ERROR,
                filename=file.filename or "unknown",
                document_id="",
                chunks=0,
                pages=0,
                error=e.message,
                error_code=e.error_code
            )
        
        except Exception as e:
            logger.exception(f"Unexpected error uploading")
            error = DocumentProcessingError(
                f"Unexpected error: {str(e)}",
                ErrorCode.PROCESSING_FAILED
            )
            return ProcessDocumentResponse(
                status=DocumentResponseStatus.ERROR,
                filename=file.filename or "unknown",
                document_id="",
                chunks=0,
                pages=0,
                error=error.message,
                error_code=error.error_code
            )

    # ============ SEARCH & MANAGEMENT ============

    async def search(self, query: str, top_k: int = 5) -> List[ChunkSearchResult]:
        """
        Search with quality filtering: overfetch → threshold → sort → trim.
        
        CHANGED: Implements robust relevance filtering strategy:
        1. Fetch more candidates than needed (15 vs 5) to find strong matches
        2. Filter by similarity threshold (0.70 default) to remove weak results
        3. Sort by score (highest first) for best ordering
        4. Trim to top_k to respect user's result limit
        5. Return empty list if nothing passes threshold (handled gracefully by API)
        
        Why overfetch (SEARCH_CANDIDATE_K > top_k):
        - Vector stores return *nearest* neighbors, not necessarily *relevant* ones
        - Top-5 raw results might all be mediocre (scores 0.45-0.60)
        - Fetching 15 candidates increases chance of finding 2-3 strong matches (0.70+)
        - After filtering weak results, we keep the best up to top_k
        
        Result: User sees 0-5 highly relevant results, never weak matches.
        """
        try:
            query_embedding = await self.embedding_service.generate_query_embedding(query)
            
            # ============= NEW: Overfetch Strategy =============
            # Fetch more candidates than we'll display (robust retrieval)
            candidate_k = max(top_k, settings.SEARCH_CANDIDATE_K)
            raw_results = await self.vector_store.search(query_embedding, candidate_k)
            
            # Filter 1: Keep only chunks from documents that still exist in DB
            # (Safety check: vector store might have stale data after deletions)
            existing_docs = await self.document_repo.list_all()
            existing_ids = {doc.id for doc in existing_docs}
            
            # Filter 2: Apply similarity threshold (remove weak matches)
            threshold = settings.SEARCH_SCORE_THRESHOLD
            filtered_results = [
                r for r in raw_results 
                if r.chunk.document_id in existing_ids and r.score >= threshold
            ]
            
            # Sort by score (descending) and trim to requested top_k
            filtered_results.sort(key=lambda r: r.score, reverse=True)
            final_results = filtered_results[:top_k]
            # ============= END: Overfetch Strategy =============
            
            # Log quality metrics for tuning/debugging
            if final_results:
                scores = [r.score for r in final_results]
                logger.info(
                    f"Search: Returned {len(final_results)}/{len(raw_results)} results "
                    f"(threshold={threshold:.2f}, scores: {min(scores):.3f}-{max(scores):.3f})"
                )
            else:
                logger.info(
                    f"Search: 0 results above threshold {threshold:.2f} "
                    f"(fetched {len(raw_results)} candidates)"
                )
            
            await self.message_repo.save_search_results(query, final_results)
            return final_results
            
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
            
            if stored_filename:
                try:
                    await self.file_storage.delete(stored_filename)
                except Exception as e:
                    logger.warning(f"File deletion failed: {e}")
            
            try:
                await self.vector_store.delete_by_document(document_id)
            except Exception as e:
                logger.warning(f"Vector deletion failed: {e}")
            
            return await self.document_repo.delete(document_id)
            
        except Exception as e:
            logger.error(f"Document deletion failed: {e}")
            return False
            
    async def list_documents(self, request: Request) -> List[DocumentsListItem]:
        """List all documents with download URLs"""
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
        
    async def get_document_with_path(
        self, document_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get document details and physical file path"""
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
        """Clear all documents, vectors, and chat history"""
        try:
            documents = await self.document_repo.list_all()
            
            # Delete physical files
            for doc in documents:
                stored_filename = doc.metadata.get("stored_filename")
                if stored_filename:
                    await self.file_storage.delete(stored_filename)
            
            # Clear vector store, database, and chat history
            return (
                await self.vector_store.clear() and 
                await self.document_repo.delete_all() and
                await self.message_repo.clear_history()  # ADD THIS LINE
            )
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
    

    # ============ DEBUG & TROUBLESHOOTNG ============

    def _save_chunks_for_debug(self, chunks: List[DocumentChunk], filename: str) -> None:
        """Save extracted chunks to text file for OCR troubleshooting."""
        debug_dir = Path("debug_ocr")
        debug_dir.mkdir(exist_ok=True)
        
        debug_file = debug_dir / f"{filename}.txt"
        
        with open(debug_file, "w", encoding="utf-8") as f:
            f.write(f"Total chunks: {len(chunks)}\n")
            f.write("=" * 80 + "\n\n")
            
            for i, chunk in enumerate(chunks, 1):
                f.write(f"CHUNK {i}\n")
                f.write(f"Page: {chunk.metadata.get('page', 'N/A')}\n")
                f.write("-" * 80 + "\n")
                f.write(chunk.content)
                f.write("\n\n" + "=" * 80 + "\n\n")
        
        logger.info(f"Debug: Saved {len(chunks)} chunks to {debug_file}")