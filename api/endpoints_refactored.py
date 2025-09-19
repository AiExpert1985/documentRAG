# api/endpoints_refactored.py
import os
import re
from typing import List, Dict, Optional, AsyncGenerator # CHANGED: Import AsyncGenerator

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, Request
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from core.interfaces import IRAGService
from database.chat_db import AsyncSessionLocal
from services.factory import ServiceFactory
from api.types import (
    SearchResponse, UploadResponse, StatusResponse, 
    DocumentsListResponse, DeleteResponse, ChatRequest
)
from utils.helpers import get_file_hash

router = APIRouter()

# Dependency injection
# CHANGED: Updated the return type hint to fix the Pylance error
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session

def get_rag_service(db: AsyncSession = Depends(get_db)) -> IRAGService:
    return ServiceFactory.create_rag_service(db)

# Utility functions
def validate_document_id(doc_id: str) -> bool:
    uuid_pattern = r'^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$'
    return bool(re.match(uuid_pattern, doc_id, re.IGNORECASE))

# API Endpoints
@router.post("/search", response_model=SearchResponse)
async def search_endpoint(
    request: ChatRequest,
    rag_service: IRAGService = Depends(get_rag_service)
) -> SearchResponse:
    if not 3 <= len(request.question) <= 2000:
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
    
    results = await rag_service.search(request.question, top_k=5)
    
    search_results = []
    for result in results:
        search_results.append({
            "document_name": result.chunk.metadata.get("document_name", "Unknown"),
            "page_number": result.chunk.metadata.get("page", 0),
            "content_snippet": (
                result.chunk.content[:300] + "..." 
                if len(result.chunk.content) > 300 
                else result.chunk.content
            ),
            "document_id": result.chunk.document_id
        })
    
    return SearchResponse(
        status="success",
        query=request.question,
        results=search_results,
        total_results=len(search_results)
    )

@router.post("/upload-pdf", response_model=UploadResponse)
async def upload_pdf(
    file: UploadFile = File(...),
    processing_strategy: Optional[str] = Form(None),
    rag_service: IRAGService = Depends(get_rag_service)
) -> UploadResponse:
    if not file.filename or not file.filename.lower().endswith(('.pdf')):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    if file.size and file.size > settings.MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail=f"File too large. Max size: {settings.MAX_FILE_SIZE // 1024 // 1024}MB")
    
    file_content = await file.read()
    file_hash = get_file_hash(file_content)
    await file.seek(0)
    
    result = await rag_service.process_document(
        file, file.filename, file_hash, processing_strategy=processing_strategy
    )
    
    if result["status"] == "success":
        result['message'] = f"Successfully processed '{result['filename']}'"
        return UploadResponse(**result)
    else:
        status_code = 400 if "exists" in result.get("error", "") else 500
        raise HTTPException(status_code=status_code, detail=result.get("error", "Processing failed"))

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
    db: AsyncSession = Depends(get_db)
) -> List[Dict]:
    from infrastructure.repositories import SQLMessageRepository
    message_repo = SQLMessageRepository(db)
    return await message_repo.get_search_history(limit=50)

@router.delete("/search-history", response_model=DeleteResponse)
async def clear_search_history(
    db: AsyncSession = Depends(get_db)
) -> DeleteResponse:
    from infrastructure.repositories import SQLMessageRepository
    message_repo = SQLMessageRepository(db)
    success = await message_repo.clear_history()
    if success:
        return DeleteResponse(
            status="success",
            message="Search history cleared successfully"
        )
    raise HTTPException(status_code=500, detail="Failed to clear search history")