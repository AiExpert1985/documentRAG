# api/endpoints_refactored.py
"""Refactored API endpoints with proper service layer"""
import os
import tempfile
import re
from typing import AsyncGenerator, List, Dict, Optional
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, Request
from fastapi.responses import HTMLResponse
from sqlalchemy.ext.asyncio import AsyncSession
from werkzeug.utils import secure_filename

from config import settings
from core.interfaces import IRAGService
from database.chat_db import AsyncSessionLocal
from infrastructure.repositories import SQLMessageRepository
from services.factory import ServiceFactory
from api.types import (
    ChatRequest, SearchResponse, UploadResponse, 
    StatusResponse, DocumentsListResponse, DeleteResponse
)
from utils.helpers import get_file_hash

router = APIRouter()

# Dependency injection
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session

def get_rag_service(db: AsyncSession = Depends(get_db)) -> IRAGService:
    """Get RAG service instance"""
    return ServiceFactory.create_rag_service(db)

# Utility functions
def validate_document_id(doc_id: str) -> bool:
    """Validate document ID format"""
    uuid_pattern = r'^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$'
    return bool(re.match(uuid_pattern, doc_id, re.IGNORECASE))

# API Endpoints
@router.post("/search", response_model=SearchResponse)
async def search_endpoint(
    request: ChatRequest,
    rag_service: IRAGService = Depends(get_rag_service)
) -> SearchResponse:
    """Search documents"""
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
    """
    Upload and process a PDF.
    
    You can optionally specify a processing_strategy in the form data:
    - **auto**: (Default) System tries to detect if OCR is needed.
    - **unstructured**: Forces direct text extraction.
    - **easyocr**: Forces OCR processing with EasyOCR.
    - **paddleocr**: Forces OCR processing with PaddleOCR.
    """
    if not file.filename or not file.filename.lower().endswith(('.pdf')):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    safe_filename = secure_filename(file.filename)
    
    if file.size and file.size > settings.MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail=f"File too large. Max size: {settings.MAX_FILE_SIZE // 1024 // 1024}MB")
    
    file_content = await file.read()
    file_hash = get_file_hash(file_content)
    
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(file_content)
            temp_path = temp_file.name
        
        result = await rag_service.process_document(
            temp_path,
            safe_filename,
            file_hash,
            processing_strategy=processing_strategy
        )
        
        if result["status"] == "success":
            # Add a message field for the response model
            result['message'] = f"Successfully processed '{result['filename']}'"
            return UploadResponse(**result)
        else:
            status_code = 400 if "exists" in result.get("error", "") or "Unsupported" in result.get("error", "") else 500
            raise HTTPException(status_code=status_code, detail=result.get("error", "Processing failed"))
            
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)

@router.delete("/documents/{document_id}", response_model=DeleteResponse)
async def delete_document(
    document_id: str,
    rag_service: IRAGService = Depends(get_rag_service)
) -> DeleteResponse:
    """Delete a document"""
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
    """Clear all documents"""
    await rag_service.clear_all()
    return DeleteResponse(
        status="success",
        message="All documents cleared successfully"
    )

@router.get("/documents", response_model=DocumentsListResponse)
async def list_documents(
    rag_service: IRAGService = Depends(get_rag_service)
) -> DocumentsListResponse:
    """List all documents"""
    documents = await rag_service.list_documents()
    return DocumentsListResponse(
        documents=[{"id": doc["id"], "filename": doc["filename"]} for doc in documents]
    )

@router.get("/status", response_model=StatusResponse)
async def get_status(
    rag_service: IRAGService = Depends(get_rag_service)
) -> StatusResponse:
    """Get system status"""
    status = await rag_service.get_status()
    return StatusResponse(**status)

@router.get("/search-history", response_model=List[Dict])
async def get_search_history(
    db: AsyncSession = Depends(get_db)
) -> List[Dict]:
    """Get recent search history"""
    message_repo = SQLMessageRepository(db)
    return await message_repo.get_search_history(limit=50)

@router.delete("/search-history", response_model=DeleteResponse)
async def clear_search_history(
    db: AsyncSession = Depends(get_db)
) -> DeleteResponse:
    """Clear all search history"""
    message_repo = SQLMessageRepository(db)
    success = await message_repo.clear_history()
    if success:
        return DeleteResponse(
            status="success",
            message="Search history cleared successfully"
        )
    raise HTTPException(status_code=500, detail="Failed to clear search history")