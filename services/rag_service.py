# services/rag_service.py
from collections import defaultdict
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from uuid import uuid4
from PIL import Image

from fastapi import UploadFile, Request, HTTPException
from pydantic import HttpUrl

from api.schemas import (
    ProcessDocumentResponse, ErrorCode, ProcessingStatus, 
    DocumentProcessingError, DocumentsListItem
)
from core.domain import DocumentResponseStatus, PageSearchResult
from core.interfaces import (
    IDocumentProcessor, IRAGService, IReranker, IVectorStore, IEmbeddingService, 
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

from utils.arabic_text import has_keyword_match

from config import settings

import shutil
import asyncio

logger = logging.getLogger(__name__)

class RAGService(IRAGService):
    def __init__(
        self,
        vector_store: IVectorStore,
        doc_processor_factory: DocumentProcessorFactory,
        embedding_service: IEmbeddingService,
        file_storage: IFileStorage,
        document_repo: IDocumentRepository,
        message_repo: IMessageRepository,
        reranker: Optional[IReranker] = None
    ):
        self.document_repo = document_repo
        self.message_repo = message_repo
        self.vector_store = vector_store
        self.doc_processor_factory = doc_processor_factory
        self.embedding_service = embedding_service
        self.file_storage = file_storage
        self.reranker = reranker  # NEW

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
        Store chunks in vector database. Progress: 80-95%.
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

        progress_store.update(doc_id, ProcessingStatus.STORING, 95,
                            "Storage complete")
    
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
        self,
        doc_id: str,
        file_path: str,
        file_type: str,
        file_hash: str,
        filename: str,
        stored_name: str
    ) -> None:
        """Background processing pipeline with page image saving and metadata persistence."""
        document: Optional[ProcessedDocument] = None
        
        async with get_session() as session:
            doc_repo = SQLDocumentRepository(session)
            
            try:
                # 1) Check duplicate
                existing = await doc_repo.get_by_hash(file_hash)
                if existing:
                    raise DocumentProcessingError(
                        "Document already exists",
                        ErrorCode.DUPLICATE_FILE
                    )
                
                # 2) Create doc record
                document = await doc_repo.create(doc_id, filename, file_hash, stored_name)

                # 3) Convert PDF → images
                processor = self.doc_processor_factory.get_processor(file_type)
                images: List[Image.Image] = await processor.load_images(file_path, file_type)

                # 4) Save page images (+ thumbnails)
                page_image_paths, page_thumbnail_paths = {}, {}
                for page_num, image in enumerate(images, 1):
                    original_rel, thumb_rel = await self.file_storage.save_page_image(
                        image=image, document_id=doc_id, page_number=page_num
                    )
                    page_image_paths[page_num] = original_rel
                    page_thumbnail_paths[page_num] = thumb_rel

                # 5) Persist metadata
                document.metadata = (document.metadata or {})
                document.metadata["page_image_paths"] = page_image_paths
                document.metadata["page_thumbnail_paths"] = page_thumbnail_paths
                await doc_repo.update_metadata(document.id, document.metadata)

                # 6) OCR → chunking (with fallback)
                try:
                    chunks = await self._extract_text_chunks(file_path, file_type, document, doc_id)
                except RuntimeError as ocr_error:
                    # Primary OCR engine failed, try fallback
                    logger.warning(f"[PROCESS] Primary OCR failed: {ocr_error}")
                    logger.info(f"[PROCESS] Attempting OCR fallback for {filename}")
                    
                    # This will be caught by _extract_text_chunks if it uses factory
                    # Or handle here if you want explicit fallback control
                    chunks = await self._extract_text_chunks_with_fallback(
                        file_path, file_type, document, doc_id
                    )

                # Attach image paths to each chunk
                for ch in chunks:
                    p = int(ch.metadata.get("page", 0))
                    ch.metadata["image_path"] = page_image_paths.get(p, "")
                    ch.metadata["thumbnail_path"] = page_thumbnail_paths.get(p, "")

                # 7) Embeddings + store
                chunks = await self._generate_embeddings(chunks, doc_id)
                await self._store_chunks(chunks, doc_id)

                # ✅ Mark as complete
                progress_store.complete(doc_id)
                logger.info(f"[PROCESS] Successfully processed {filename}")
            
            except DocumentProcessingError as e:
                logger.error(f"[PROCESS] Processing failed for '{filename}': {e.message}")
                progress_store.fail(doc_id, e.message, e.error_code)
                await self._cleanup_on_failure(
                    document.id if document else None,
                    stored_name, 
                    doc_id,
                    doc_repo
                )
            
            except Exception as e:
                logger.exception(f"[PROCESS] Unexpected error processing '{filename}'")
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

    # ============ RERANK ============

    def _should_rerank(self, scores: List[float]) -> bool:
        """
        Determine if reranking is needed based on score distribution.
        
        Skip reranking for:
        - Very strong matches (>0.75) - already confident
        - Very weak matches (<0.55) - won't improve much
        
        Only rerank borderline cases (0.55-0.75) where neural precision helps.
        """
        if not scores:
            return False
        
        top_score = max(scores)
        return settings.RERANK_GATE_LOW <= top_score <= settings.RERANK_GATE_HIGH

    async def _maybe_rerank(
        self, 
        query: str, 
        candidates: List[ChunkSearchResult]
    ) -> List[ChunkSearchResult]:
        """
        Conditionally rerank results with cross-encoder.
        
        Gates reranking by score and caps candidates to prevent latency spikes.
        Returns reranked top candidates + remaining results in original order.
        """
        if not self.reranker:
            return candidates
        
        # Check if reranking is needed
        scores = [c.score for c in candidates]
        if not self._should_rerank(scores):
            logger.info(f"[RERANK] Skipping (top score: {max(scores):.3f})")
            return candidates
        
        # Cap candidates to prevent slow cross-encoder on large sets
        cap = min(len(candidates), settings.RERANK_CANDIDATE_CAP)
        capped = candidates[:cap]
        
        # Rerank top candidates
        reranked = await self.reranker.rerank(query, capped, top_k=cap)
        
        # Preserve non-reranked results
        capped_ids = {c.chunk.id for c in capped}
        rest = [c for c in candidates if c.chunk.id not in capped_ids]
        
        logger.info(f"[RERANK] Reranked {len(capped)} candidates")
        return reranked + rest
    
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

    async def search_chunks(self, query: str, top_k: int = 5) -> List[ChunkSearchResult]:
        """
        Hybrid search: Dense → Lexical → Rerank.
        
        Pipeline:
        1. Dense retrieval (broad recall)
        2. Basic filters (existence + threshold)
        3. Lexical gate (keyword matching)
        4. Neural rerank (semantic precision)
        """
        try:
            # Stage 1: Dense retrieval
            query_embedding = await self.embedding_service.generate_query_embedding(query)
            candidate_k = max(top_k * 2, settings.SEARCH_CANDIDATE_K)
            raw_results = await self.vector_store.search(query_embedding, candidate_k)
            
            # Stage 2: Basic filters
            if raw_results:
                doc_ids = {r.chunk.document_id for r in raw_results}
                existing_ids = await self.document_repo.exists_bulk(list(doc_ids))
            else:
                existing_ids = set()
            
            threshold = settings.SEARCH_SCORE_THRESHOLD
            filtered_results = [
                r for r in raw_results
                if r.chunk.document_id in existing_ids and r.score >= threshold
            ]
            
            if not filtered_results:
                await self.message_repo.save_search_results(query, [])
                return []
            
            # Stage 3: Lexical gate (robust Arabic handling)
            candidates = filtered_results
            if settings.LEXICAL_GATE_ENABLED:
                min_keywords = settings.LEXICAL_MIN_KEYWORDS
                candidates_with_keywords = [
                    r for r in filtered_results
                    if has_keyword_match(query, r.chunk.content, min_keywords)
                ]
                if candidates_with_keywords:
                    candidates = candidates_with_keywords
            
            # Stage 4: Neural reranker (with gating)
            final_results = candidates
            if settings.RERANK_ENABLED and self.reranker:
                # Smart reranking: only rerank borderline scores
                reranked = await self._maybe_rerank(query, candidates)
                
                # Apply threshold and cap
                rerank_threshold = settings.RERANK_SCORE_THRESHOLD
                final_results = [
                    r for r in reranked 
                    if r.score >= rerank_threshold
                ][:top_k]
            else:
                candidates.sort(key=lambda r: r.score, reverse=True)
                final_results = candidates[:top_k]
            
            # Logging & save
            if final_results:
                scores = [r.score for r in final_results]
                logger.info(
                    f"[SEARCH] raw={len(raw_results)} filtered={len(filtered_results)} "
                    f"lexical={len(candidates)} final={len(final_results)} | "
                    f"scores={min(scores):.3f}-{max(scores):.3f}"
                )
            
            await self.message_repo.save_search_results(query, final_results)
            return final_results
            
        except Exception as e:
            logger.error(f"[SEARCH] Failed: {e}", exc_info=True)
            return []

 
 

    async def delete_document(self, document_id: str) -> bool:
        try:
            doc = await self.document_repo.get_by_id(document_id)
            if not doc:
                return False

            # Remove original file (if present in metadata)
            stored_filename = doc.metadata.get("stored_filename") if doc.metadata else None
            if stored_filename:
                try:
                    await self.file_storage.delete(stored_filename)
                except Exception as e:
                    logger.warning(f"File deletion failed: {e}")

            # Remove vectors (best-effort)
            try:
                await self.vector_store.delete_by_document(document_id)
            except Exception as e:
                logger.warning(f"Vector deletion failed: {e}")

            # Remove page images dir for this doc
            page_dir = Path(settings.UPLOADS_DIR) / "page_images" / document_id
            try:
                await asyncio.to_thread(shutil.rmtree, page_dir, ignore_errors=True)
            except Exception as e:
                logger.warning(f"Page image folder deletion failed: {e}")

            # Finally DB record
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
        
    async def get_document_with_path(self, document_id: str):
        """
        Return original filename + RELATIVE stored filename.
        Endpoint will resolve safely.
        """
        doc = await self.document_repo.get_by_id(document_id)
        if not doc:
            return None

        stored_filename = (doc.metadata or {}).get("stored_filename")
        if not stored_filename:
            return None

        return {
            "original_filename": doc.filename,
            "path": stored_filename,  # relative
        }
    

    async def clear_all(self) -> bool:
        """Clear all documents, vectors, chat history, and page images."""
        try:
            docs = await self.document_repo.list_all()

            # Delete originals (best-effort)
            for d in docs:
                stored_filename = d.metadata.get("stored_filename") if d.metadata else None
                if stored_filename:
                    try:
                        await self.file_storage.delete(stored_filename)
                    except Exception as e:
                        logger.warning(f"Delete original failed: {e}")

            # Remove entire page_images tree
            base = Path(settings.UPLOADS_DIR) / "page_images"
            try:
                await asyncio.to_thread(shutil.rmtree, base, ignore_errors=True)
            except Exception as e:
                logger.warning(f"Failed to remove page_images dir: {e}")

            # Clear vectors, DB, and chat history
            ok = await self.vector_store.clear()
            ok = ok and await self.document_repo.delete_all()
            ok = ok and await self.message_repo.clear_history()
            return ok
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

        

    async def search_pages(self, query: str, top_k: int = 5) -> List[PageSearchResult]:
        """Search and return page-level results with images (UI-ready)."""
        try:
            # Overfetch chunks, then aggregate to pages
            chunk_results = await self.search_chunks(query, top_k=top_k * 3)
            if not chunk_results:
                logger.info(f"[SEARCH-PAGES] No chunks for: {query[:60]}")
                return []

            page_results = await self._aggregate_chunks_to_pages(query, chunk_results, top_k)

            return page_results

        except Exception as e:
            logger.error(f"[SEARCH-PAGES] Failed: {e}", exc_info=True)
            return []

    async def search(self, query: str, top_k: int = 5) -> List[PageSearchResult]:
        """Alias: default to page-level results."""
        return await self.search_pages(query, top_k)

    async def _aggregate_chunks_to_pages(
        self,
        query: str,
        chunk_results: List[ChunkSearchResult],
        top_k: int
    ) -> List[PageSearchResult]:
        """Group chunks by (document_id, page), score with MaxP, build highlights, and attach image URLs."""
        groups = defaultdict(list)
        for r in chunk_results:
            key = (r.chunk.document_id, int(r.chunk.metadata.get("page", 0)))
            groups[key].append(r)

        pages: List[PageSearchResult] = []

        for (doc_id, page_num), items in groups.items():
            if not items:
                continue

            # Scoring: MaxP (primary)
            max_score = max((it.score for it in items), default=0.0)

            # Highlights (lexical-first, fallback to excerpt)
            highlights = self._extract_highlights(query, items, max_count=3)
            if not highlights:
                best = max(items, key=lambda x: x.score)
                c = best.chunk.content
                highlights = [c[:150] + ("..." if len(c) > 150 else "")]

            # Image paths (already attached during processing)
            image_path = items[0].chunk.metadata.get("image_path", "")
            thumb_path = items[0].chunk.metadata.get("thumbnail_path", "")

            if image_path:
                image_url = f"/page-image/{doc_id}/{page_num}"
                thumbnail_url = f"/page-image/{doc_id}/{page_num}?size=thumbnail"
            else:
                image_url = ""
                thumbnail_url = ""

            doc_name = items[0].chunk.metadata.get("document_name", "Unknown")

            pages.append(PageSearchResult(
                document_id=doc_id,
                document_name=doc_name,
                page_number=page_num,
                score=max_score,
                chunk_count=len(items),
                image_url=image_url,
                thumbnail_url=thumbnail_url,
                highlights=highlights,
                download_url=f"/download/{doc_id}",
            ))

        pages.sort(key=lambda p: (p.score, p.chunk_count), reverse=True)
        return pages[:top_k]


    def _extract_highlights(
        self,
        query: str,
        items: List[ChunkSearchResult],
        max_count: int = 3
    ) -> List[str]:
        """Keyword-aware highlights with Arabic normalization; fallback handled by caller."""
        try:
            from utils.arabic_text import extract_keywords, normalize_arabic
        except Exception:
            # if not available, return top chunk excerpt
            if not items:
                return []
            best = max(items, key=lambda x: x.score)
            return [(best.chunk.content[:150] + "...") if len(best.chunk.content) > 150 else best.chunk.content]

        kws = extract_keywords(query)
        if not kws:
            # no keywords → just return top chunk excerpt
            if not items:
                return []
            best = max(items, key=lambda x: x.score)
            return [(best.chunk.content[:150] + "...") if len(best.chunk.content) > 150 else best.chunk.content]

        out: List[str] = []
        # take top 3 chunks by score
        for it in sorted(items, key=lambda x: x.score, reverse=True)[:3]:
            text = it.chunk.content
            norm = normalize_arabic(text)
            hit = None
            for kw in kws:
                if kw in norm:
                    hit = kw
                    break
            if hit:
                words = text.split()
                # approximate location by scanning normalized tokens
                for i, w in enumerate(words):
                    if hit in normalize_arabic(w):
                        start = max(0, i - 10)
                        end = min(len(words), i + 11)
                        snippet = " ".join(words[start:end])
                        if len(snippet) > 150:
                            snippet = snippet[:147] + "..."
                        if start > 0:
                            snippet = "..." + snippet
                        if end < len(words):
                            snippet = snippet + "..."
                        out.append(snippet)
                        break
            if len(out) >= max_count:
                break

        return out[:max_count]
    
    async def _extract_text_chunks_with_fallback(
        self, file_path: str, file_type: str, document: ProcessedDocument, doc_id: str
    ) -> List[DocumentChunk]:
        """
        Extract text with automatic OCR engine fallback.
        Tries primary engine, then fallback if primary fails.
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
        
        # Try primary processor
        try:
            processor = self.doc_processor_factory.get_processor(file_type)
            chunks = await processor.process(file_path, file_type, update_page_progress)
        except RuntimeError as e:
            logger.warning(f"[OCR] Primary engine failed: {e}, trying fallback...")
            
            # Try fallback processor
            try:
                processor = self.doc_processor_factory.get_fallback_processor(file_type)
                chunks = await processor.process(file_path, file_type, update_page_progress)
            except RuntimeError as fallback_error:
                raise DocumentProcessingError(
                    f"All OCR engines failed. Last error: {fallback_error}",
                    ErrorCode.PROCESSING_FAILED
                )
        
        if not chunks:
            raise DocumentProcessingError(
                "No content extracted from document", 
                ErrorCode.NO_TEXT_FOUND
            )
        
        # Attach metadata
        for chunk in chunks:
            chunk.document_id = document.id
            chunk.metadata.update({
                "document_id": document.id, 
                "document_name": document.filename
            })
        
        # Debug save
        self._save_chunks_for_debug(chunks, document.filename)
        
        return chunks