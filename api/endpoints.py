# api/endpoints.py
import os
from typing import List, Dict

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Request
from fastapi.responses import FileResponse

from config import settings
from core.interfaces import IMessageRepository, IRAGService
from services.factory import get_message_repository, get_rag_service
from api.schemas import (
    ProcessDocumentResponse, SearchResponse, StatusResponse, 
    DocumentsListResponse, DeleteResponse, ChatRequest
)
from utils.helpers import validate_document_id


router = APIRouter()

@router.post("/upload-document", response_model=ProcessDocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    rag_service: IRAGService = Depends(get_rag_service)
) -> ProcessDocumentResponse:
    return await rag_service.process_document(file) 

@router.post("/search", response_model=SearchResponse)
async def search_endpoint(
    request: Request, 
    chat_request: ChatRequest, 
    rag_service: IRAGService = Depends(get_rag_service)
) -> SearchResponse:
    if not 3 <= len(chat_request.question) <= 2000:
        raise HTTPException(
            status_code=422,
            detail="Search query must be between 3 and 2000 characters"
        )
    
    status = await rag_service.get_status()
    if not status["ready_for_queries"]:
        raise HTTPException(
            status_code=400,
            detail="Please upload a document first"
        )
    
    results = await rag_service.search(chat_request.question, top_k=settings.DEFAULT_SEARCH_RESULTS)
    
    base_url = str(request.base_url)
    search_results = []
    
    for result in results:
        content = result.chunk.content
        content_snippet = (
            content[:settings.SNIPPET_LENGTH] + "..." 
            if len(content) > settings.SNIPPET_LENGTH 
            else content
        )
        
        search_results.append({
            "document_name": result.chunk.metadata.get("document_name", "Unknown"),
            "page_number": result.chunk.metadata.get("page", 0),
            "content_snippet": content_snippet,
            "document_id": result.chunk.document_id,
            "download_url": f"{base_url}download/{result.chunk.document_id}"
        })
    
    return SearchResponse(
        status="success",
        query=chat_request.question,
        results=search_results,
        total_results=len(search_results)
    )


@router.get("/download/{document_id}")
async def download_document(
    document_id: str,
    rag_service: IRAGService = Depends(get_rag_service)
):
    """Download the original document file."""
    if not validate_document_id(document_id):
        raise HTTPException(status_code=422, detail="Invalid document ID format")
    
    doc_details = await rag_service.get_document_with_path(document_id)
    
    if not doc_details or not os.path.exists(doc_details["path"]):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(
        path=doc_details["path"],
        filename=doc_details["original_filename"],
        media_type='application/octet-stream'
    )

@router.delete("/documents/{document_id}", response_model=DeleteResponse)
async def delete_document(
    document_id: str,
    rag_service: IRAGService = Depends(get_rag_service)
) -> DeleteResponse:
    if not validate_document_id(document_id):
        raise HTTPException(
            status_code=422,
            detail="Invalid document ID format"
        )
    
    success = await rag_service.delete_document(document_id)
    if not success:
        raise HTTPException(
            status_code=404,
            detail="Document not found"
        )
    
    return DeleteResponse(
        status="success",
        message="Document deleted successfully"
    )

@router.delete("/documents", response_model=DeleteResponse)
async def clear_all_documents(
    rag_service: IRAGService = Depends(get_rag_service)
) -> DeleteResponse:
    await rag_service.clear_all()
    return DeleteResponse(
        status="success",
        message="All documents cleared successfully"
    )

@router.get("/documents", response_model=DocumentsListResponse)
async def list_documents(
    request: Request,
    rag_service: IRAGService = Depends(get_rag_service)
) -> DocumentsListResponse:
    documents = await rag_service.list_documents(request)
    return DocumentsListResponse(documents=documents)

@router.get("/status", response_model=StatusResponse)
async def get_status(
    rag_service: IRAGService = Depends(get_rag_service)
) -> StatusResponse:
    status = await rag_service.get_status()
    return StatusResponse(**status)

@router.get("/search-history", response_model=List[Dict])
async def get_search_history(
    message_repo: IMessageRepository = Depends(get_message_repository) 
) -> List[Dict]:
    return await message_repo.get_search_history(limit=50)

@router.delete("/search-history", response_model=DeleteResponse)
async def clear_search_history(
    message_repo: IMessageRepository = Depends(get_message_repository)
) -> DeleteResponse:
    success = await message_repo.clear_history()
    if success:
        return DeleteResponse(
            status="success",
            message="Search history cleared successfully"
        )
    raise HTTPException(status_code=500, detail="Failed to clear search history")


@router.get("/config")
async def get_config():
    return {
        "allowed_extensions": settings.ALLOWED_FILE_EXTENSIONS,
        "max_file_size_mb": settings.MAX_FILE_SIZE // (1024 * 1024)
    }