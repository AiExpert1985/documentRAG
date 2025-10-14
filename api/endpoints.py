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
import os
from pathlib import Path
from typing import List, Dict, Literal

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
    results = [
        PageSearchResultItem(
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
        for p in pages
    ]

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
    """
    Serve original PNG or thumbnail WebP page image.
    """
    if not validate_document_id(document_id):
        raise HTTPException(status_code=422, detail="Invalid document ID format")

    doc = await rag_service.document_repo.get_by_id(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    if size == "thumbnail":
        paths = doc.metadata.get("page_thumbnail_paths", {})
        media_type = "image/webp"
    else:
        paths = doc.metadata.get("page_image_paths", {})
        media_type = "image/png"

    if page_number not in paths:
        raise HTTPException(status_code=404, detail=f"Page {page_number} not found")

    rel = paths[page_number]
    full = _safe_file_path(settings.UPLOADS_DIR, rel)
    if not full.exists():
        raise HTTPException(status_code=404, detail="Image file not found")

    return FileResponse(
        path=str(full),
        media_type=media_type,
        filename=f"page_{page_number}.{'webp' if size == 'thumbnail' else 'png'}",
        headers={
            "Cache-Control": "public, max-age=31536000, immutable",
            "ETag": f'"{document_id}-{page_number}-{size}"',
        },
    )


# ---------- Download original document (uses safe path) ----------
@router.get("/download/{document_id}")
async def download_document(
    document_id: str, rag_service: IRAGService = Depends(get_rag_service)
):
    """Download the original document file."""
    if not validate_document_id(document_id):
        raise HTTPException(status_code=422, detail="Invalid document ID format")

    # Use service helper to get path + original filename
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
