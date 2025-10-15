# api/endpoints.py
"""
API endpoints for document search system.

SECURITY NOTE (Phase 0 - MVP):
==================================
These endpoints currently have NO AUTHENTICATION for pilot testing.
This is an intentional decision documented in SECURITY_POLICY.md.

Before production deployment:
1. Add authentication middleware
2. Implement role-based access control
3. Add document-level permissions
4. Enable audit logging
==================================
"""

from fastapi import APIRouter, HTTPException, Response, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from pathlib import Path
from typing import Dict, Any, List, Tuple

from database.session import get_db
from infrastructure.repositories import SQLDocumentRepository
from utils.highlight_token import verify
from infrastructure.image_utils import ImageHighlighter
from config import settings
import os
from pathlib import Path
from typing import List, Dict, Literal
from utils.highlight_token import sign

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Request
from fastapi.responses import FileResponse

from config import settings
from core.interfaces import IMessageRepository, IRAGService
from services.factory import get_message_repository, get_rag_service
from api.schemas import (
    ProcessDocumentResponse,
    SearchResponse,
    StatusResponse,
    DocumentsListResponse,
    DeleteResponse,
    ChatRequest,
    PageSearchResponse,
    PageSearchResultItem,
    ProcessingProgress,
)
from utils.common import validate_document_id
from infrastructure.progress_store import progress_store

router = APIRouter()


# ---------- Helper: Safe file path join (prevents path traversal) ----------
def _safe_file_path(base_dir: str, relative_path: str) -> Path:
    """
    Safely join base directory with relative path.
    Ensures resolved path stays within base directory.
    """
    base = Path(base_dir).resolve()
    full = (base / relative_path).resolve()
    try:
        full.relative_to(base)
    except ValueError:
        # Escaped the base dir -> block
        raise HTTPException(status_code=400, detail="Invalid file path")
    return full


# ---------- Upload ----------
@router.post("/upload-document", response_model=ProcessDocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    rag_service: IRAGService = Depends(get_rag_service),
) -> ProcessDocumentResponse:
    return await rag_service.process_document(file)


# ---------- Search (CHUNKS) - keep existing behavior for LLM/compat ----------
@router.post("/search", response_model=SearchResponse)
async def search_endpoint(
    request: Request,
    chat_request: ChatRequest,
    rag_service: IRAGService = Depends(get_rag_service),
) -> SearchResponse:
    if not 3 <= len(chat_request.question) <= 2000:
        raise HTTPException(
            status_code=422, detail="Search query must be between 3 and 2000 characters"
        )

    status = await rag_service.get_status()
    if not status["ready_for_queries"]:
        raise HTTPException(status_code=400, detail="Please upload a document first")

    # IMPORTANT: Use chunk-level search to preserve old behavior
    results = await rag_service.search_chunks(
        chat_request.question, top_k=settings.DEFAULT_SEARCH_RESULTS
    )

    base_url = str(request.base_url).rstrip("/")
    search_results = []

    for result in results:
        content = result.chunk.content
        snippet = (
            content[: settings.SNIPPET_LENGTH] + "..."
            if len(content) > settings.SNIPPET_LENGTH
            else content
        )
        search_results.append(
            {
                "document_name": result.chunk.metadata.get("document_name", "Unknown"),
                "page_number": result.chunk.metadata.get("page", 0),
                "content_snippet": snippet,
                "document_id": result.chunk.document_id,
                "download_url": f"{base_url}/download/{result.chunk.document_id}",
            }
        )

    return SearchResponse(
        status="success",
        query=chat_request.question,
        results=search_results,
        total_results=len(search_results),
    )


# ---------- Search (PAGES) - new endpoint for UI with images/highlights ----------
@router.post("/search-pages", response_model=PageSearchResponse)
async def search_pages_endpoint(
    request: Request,
    chat_request: ChatRequest,
    rag_service: IRAGService = Depends(get_rag_service),
) -> PageSearchResponse:
    if not 3 <= len(chat_request.question) <= 2000:
        raise HTTPException(
            status_code=422, detail="Search query must be between 3 and 2000 characters"
        )

    status = await rag_service.get_status()
    if not status["ready_for_queries"]:
        raise HTTPException(status_code=400, detail="Please upload a document first")

    pages = await rag_service.search_pages(chat_request.question, top_k=settings.TOP_K)

    base_url = str(request.base_url).rstrip("/")
    results: List[PageSearchResultItem] = []
    base_url = str(request.base_url).rstrip("/")

    for p in pages:
        item = PageSearchResultItem(
            document_id=p.document_id,
            document_name=p.document_name,
            page_number=p.page_number,
            score=round(p.score, 3),
            chunk_count=p.chunk_count,
            image_url=(base_url + p.image_url) if p.image_url else "",
            thumbnail_url=(base_url + p.thumbnail_url) if p.thumbnail_url else "",
            highlights=p.highlights,
            download_url=(base_url + p.download_url) if p.download_url else "",
        )

        # 🔶 NEW: load page segments from document metadata -> build token
        try:
            doc = await rag_service.document_repo.get_by_id(p.document_id)
            meta = (doc.metadata or {}) if doc else {}
            page_segments = (meta.get("segments") or {}).get(str(p.page_number), [])
            seg_ids = [s.get("segment_id") for s in page_segments if s.get("segment_id")]
            if seg_ids:
                item.segment_ids = seg_ids[: settings.HIGHLIGHT_MAX_REGIONS]
                item.highlight_token = sign({
                    "doc_id": p.document_id,
                    "page_index": p.page_number,
                    "segment_ids": item.segment_ids,
                    "style_id": settings.HIGHLIGHT_STYLE_ID,
                    "source_version": "v1"
                }, exp_seconds=120)
        except Exception:
            # If anything fails here, we just send the basic item (no highlights)
            pass

        results.append(item)

    return PageSearchResponse(
        status="success",
        query=chat_request.question,
        results=results,
        total_results=len(results),
    )


# ---------- Serve page image (original or thumbnail) ----------
@router.get("/page-image/{document_id}/{page_number}")
async def get_page_image(
    document_id: str,
    page_number: int,
    size: Literal["original", "thumbnail"] = "original",
    rag_service: IRAGService = Depends(get_rag_service),
):
    if not validate_document_id(document_id):
        raise HTTPException(status_code=422, detail="Invalid document ID format")

    # Confirm doc exists (same behavior as before)
    doc = await rag_service.document_repo.get_by_id(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    suffix = "_thumb.webp" if size == "thumbnail" else ".png"
    rel = f"page_images/{document_id}/page_{page_number:03d}{suffix}"
    full = _safe_file_path(settings.UPLOADS_DIR, rel)

    if not full.exists():
        raise HTTPException(status_code=404, detail="Image file not found")

    media_type = "image/webp" if size == "thumbnail" else "image/png"
    return FileResponse(
        path=str(full),
        media_type=media_type,
        filename=Path(rel).name,
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )

# ---------- Download original document (uses safe path) ----------
@router.get("/download/{document_id}")
async def download_document(
    document_id: str, rag_service: IRAGService = Depends(get_rag_service)
):
    if not validate_document_id(document_id):
        raise HTTPException(status_code=422, detail="Invalid document ID format")

    doc_details = await rag_service.get_document_with_path(document_id)
    if not doc_details:
        raise HTTPException(status_code=404, detail="Document not found")

    full = _safe_file_path(settings.UPLOADS_DIR, doc_details["path"])
    if not full.exists():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(
        path=str(full),
        filename=doc_details["original_filename"],
        media_type="application/octet-stream",
    )


# ---------- Delete one document (keep) ----------
@router.delete("/documents/{document_id}", response_model=DeleteResponse)
async def delete_document(
    document_id: str, rag_service: IRAGService = Depends(get_rag_service)
) -> DeleteResponse:
    if not validate_document_id(document_id):
        raise HTTPException(status_code=422, detail="Invalid document ID format")

    success = await rag_service.delete_document(document_id)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found")

    return DeleteResponse(status="success", message="Document deleted successfully")


# ---------- Clear all documents (keep) ----------
@router.delete("/documents", response_model=DeleteResponse)
async def clear_all_documents(
    rag_service: IRAGService = Depends(get_rag_service),
) -> DeleteResponse:
    await rag_service.clear_all()
    return DeleteResponse(status="success", message="All documents cleared successfully")


# ---------- List documents (keep) ----------
@router.get("/documents", response_model=DocumentsListResponse)
async def list_documents(
    request: Request, rag_service: IRAGService = Depends(get_rag_service)
) -> DocumentsListResponse:
    documents = await rag_service.list_documents(request)
    return DocumentsListResponse(documents=documents)


# ---------- Service status (keep) ----------
@router.get("/status", response_model=StatusResponse)
async def get_status(rag_service: IRAGService = Depends(get_rag_service)) -> StatusResponse:
    status = await rag_service.get_status()
    return StatusResponse(**status)


# ---------- Search history (keep) ----------
@router.get("/search-history", response_model=List[Dict])
async def get_search_history(
    message_repo: IMessageRepository = Depends(get_message_repository),
) -> List[Dict]:
    return await message_repo.get_search_history(limit=50)


@router.delete("/search-history", response_model=DeleteResponse)
async def clear_search_history(
    message_repo: IMessageRepository = Depends(get_message_repository),
) -> DeleteResponse:
    success = await message_repo.clear_history()
    if success:
        return DeleteResponse(status="success", message="Search history cleared successfully")
    raise HTTPException(status_code=500, detail="Failed to clear search history")


# ---------- Config (keep) ----------
@router.get("/config")
async def get_config():
    return {
        "allowed_extensions": settings.ALLOWED_FILE_EXTENSIONS,
        "max_file_size_mb": settings.MAX_FILE_SIZE // (1024 * 1024),
    }


# ---------- Processing progress (keep) ----------
@router.get("/processing-status/{document_id}", response_model=ProcessingProgress)
async def get_processing_status(document_id: str) -> ProcessingProgress:
    """Get real-time processing progress for a document"""
    progress = progress_store.get(document_id)
    if not progress:
        raise HTTPException(status_code=404, detail="No processing status found for this document")

    return ProcessingProgress(document_id=document_id, **progress)


# ---------- Health Check ----------
@router.get("/health")
async def health_check(rag_service: IRAGService = Depends(get_rag_service)):
    """
    System health check endpoint.
    
    Returns:
        - status: healthy/unhealthy
        - chunks_indexed: Number of chunks in vector store
        - reranker_loaded: Whether reranker model is available
        - ocr_engine: Currently configured OCR backend
        - timestamp: When check was performed
    """
    try:
        from datetime import datetime
        
        # Check vector store
        count = await rag_service.vector_store.count()
        
        # Check reranker availability
        reranker_loaded = False
        if rag_service.reranker:
            reranker_loaded = bool(getattr(rag_service.reranker, "_model", None))
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "chunks_indexed": count,
            "reranker_loaded": reranker_loaded,
            "ocr_engine": settings.OCR_ENGINE,
            "vector_store": settings.VECTOR_STORE_TYPE,
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "error": str(e)
        }
    


# @router.get("/page-image/highlighted/{token}")
# async def get_highlighted_image(token: str, db: AsyncSession = Depends(get_session)):
#     if not settings.ENABLE_HIGHLIGHT_PREVIEW:
#         raise HTTPException(status_code=404, detail="Disabled")

#     try:
#         payload = verify(token)
#     except ValueError:
#         raise HTTPException(status_code=401, detail="Invalid or expired token")

#     doc_id = payload.get("doc_id"); page_index = int(payload.get("page_index", -1))
#     seg_ids = payload.get("segment_ids") or []; style_id = payload.get("style_id", settings.HIGHLIGHT_STYLE_ID)
#     if not doc_id or page_index < 0:
#         raise HTTPException(status_code=400, detail="Invalid request")

#     repo = SQLDocumentRepository(db)
#     doc = await repo.get_by_id(doc_id)
#     if not doc:
#         raise HTTPException(status_code=404, detail="Document not found")

#     meta: Dict[str, Any] = doc.metadata or {}
#     rel = (meta.get("page_image_paths") or {}).get(str(page_index))
#     if not rel:
#         raise HTTPException(status_code=404, detail="Page image not found")

#     image_path = Path(settings.UPLOADS_DIR)/rel
#     if not image_path.exists():
#         raise HTTPException(status_code=404, detail="Page image file missing")

#     page_segments: List[Dict[str, Any]] = (meta.get("segments") or {}).get(str(page_index), [])
#     if not page_segments:
#         return Response(content=image_path.read_bytes(), media_type="image/webp",
#                         headers={"X-Highlight-Status":"missing_boxes"})

#     wanted=set(seg_ids) if seg_ids else set()
#     bboxes=[]
#     for seg in page_segments:
#         if wanted and seg.get("segment_id") not in wanted:
#             continue
#         if seg.get("bbox"):
#             bboxes.append(tuple(float(x) for x in seg["bbox"]))

#     if not bboxes:
#         return Response(content=image_path.read_bytes(), media_type="image/webp",
#                         headers={"X-Highlight-Status":"missing_boxes"})

#     try:
#         img = ImageHighlighter.draw_highlights(
#             image_path=str(image_path),
#             normalized_bboxes=bboxes,
#             style_id=style_id,
#             max_regions=settings.HIGHLIGHT_MAX_REGIONS,
#             timeout_sec=settings.HIGHLIGHT_TIMEOUT_SEC,
#             fmt="WEBP",
#         )
#         return Response(content=img, media_type="image/webp", headers={"X-Highlight-Status":"ok"})
#     except Exception:
#         return Response(content=image_path.read_bytes(), media_type="image/webp",
#                         headers={"X-Highlight-Status":"error"})


@router.get("/page-image/highlighted/{document_id}/{page_number}")
async def get_highlighted_image_direct(
    document_id: str,
    page_number: int,
    db: AsyncSession = Depends(get_db),   # ✅ inject an AsyncSession
):
    repo = SQLDocumentRepository(db)
    doc = await repo.get_by_id(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    meta = doc.metadata or {}
    # Paths are stored with string keys
    page_images = (meta.get("page_image_paths") or {})
    rel = page_images.get(str(page_number))
    if not rel:
        raise HTTPException(status_code=404, detail="Page image not found")

    # Normalized highlight boxes per page (saved by RAGService)
    segments_by_page = (meta.get("segments") or {})
    segs = segments_by_page.get(str(page_number), [])  # list of {"bbox": [x,y,w,h], ...}

    # Collect normalized bboxes
    bboxes = []
    for seg in segs[: settings.HIGHLIGHT_MAX_REGIONS]:
        bbox = seg.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            # ensure float tuple
            x, y, w, h = bbox
            bboxes.append((float(x), float(y), float(w), float(h)))

    abs_path = Path(settings.UPLOADS_DIR) / rel
    if not abs_path.exists():
        raise HTTPException(status_code=404, detail="Image file missing")

    # Draw and return WEBP (fast + small)
    img_bytes = ImageHighlighter.draw_highlights(
        str(abs_path),
        bboxes,
        max_regions=settings.HIGHLIGHT_MAX_REGIONS,
        timeout_sec=2,
        fmt="WEBP",
    )
    return Response(content=img_bytes, media_type="image/webp")