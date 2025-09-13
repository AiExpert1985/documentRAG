# api/endpoints.py
import logging
import tempfile
import os
import re
from typing import List, Dict, AsyncGenerator
from werkzeug.utils import secure_filename

from sqlalchemy import delete, select
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Request
from fastapi.responses import HTMLResponse
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from services.rag_service import RAGService
from database.chat_db import get_file_hash, check_file_hash, AsyncSessionLocal, Message, Document
from api.types import ChatRequest, SearchResponse, UploadResponse, StatusResponse, DocumentsListResponse, DeleteResponse, Message as MessageType, DocumentsListItem

logger = logging.getLogger(settings.LOGGER_NAME)
router = APIRouter()

# Dependency Injection and Chat History Utilities remain the same...
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session
def get_rag_service(request: Request) -> RAGService:
    return request.app.state.rag_service
async def save_search_query(db: AsyncSession, query: str, results_count: int):
    try:
        new_message = Message(sender="search", content=f"Query: {query} | Results: {results_count}")
        db.add(new_message)
        await db.commit()
    except Exception as e:
        logger.error(f"Failed to save search query: {e}")
        await db.rollback()
async def get_search_history(db: AsyncSession, limit: int = 10) -> List[Dict]:
    try:
        result = await db.execute(select(Message).where(Message.sender == "search").order_by(Message.timestamp.desc()).limit(limit))
        messages = result.scalars().all()
        return [{"query": msg.content, "timestamp": msg.timestamp} for msg in reversed(messages)]
    except Exception as e:
        logger.error(f"Failed to get search history: {e}")
        return []

# --- Updated Search Endpoint ---
@router.post("/search", response_model=SearchResponse)
async def search_endpoint(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
    rag_service: RAGService = Depends(get_rag_service)
) -> SearchResponse:
    if not 3 <= len(request.question) <= 2000:
        raise HTTPException(status_code=422, detail="Search query must be between 3 and 2000 characters.")
    if not await rag_service.has_documents(db):
        raise HTTPException(status_code=400, detail="Please upload a PDF document first.")
    
    try:
        relevant_chunks = await rag_service.retrieve_chunks(db, request.question, top_k=5)
        if not relevant_chunks:
            await save_search_query(db, request.question, 0)
            return SearchResponse(status="success", query=request.question, results=[], total_results=0)
        
        # Format results for frontend
        search_results = []
        for chunk in relevant_chunks:
            search_results.append({
                "document_name": chunk['metadata']['document_name'],
                "page_number": chunk['metadata']['page'],
                "content_snippet": chunk['content'][:300] + "..." if len(chunk['content']) > 300 else chunk['content'],
                "document_id": chunk['metadata']['document_id']
            })
        
        await save_search_query(db, request.question, len(search_results))
        return SearchResponse(
            status="success", 
            query=request.question, 
            results=search_results, 
            total_results=len(search_results)
        )
        
    except Exception as e:
        logger.error(f"Search endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal processing error occurred.")

# Add document download endpoint
@router.get("/documents/{document_id}/download")
async def download_document(
    document_id: str,
    db: AsyncSession = Depends(get_db)
):
    try:
        # Get document from database
        document = await db.get(Document, document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # For now, return document info - you'll need to implement actual file serving
        return {"document_name": document.filename, "document_id": document.id}
        
    except Exception as e:
        logger.error(f"Document download error: {e}")
        raise HTTPException(status_code=500, detail="Failed to download document")

# Upload endpoint remains the same
@router.post("/upload-pdf", response_model=UploadResponse)
async def upload_pdf(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    rag_service: RAGService = Depends(get_rag_service)
) -> UploadResponse:
    if not file.filename or not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    safe_filename = secure_filename(file.filename)
    
    if file.size and file.size > settings.MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail=f"File too large. Maximum size is {settings.MAX_FILE_SIZE // 1024 // 1024}MB.")
    
    file_content = await file.read()
    file_hash = get_file_hash(file_content)
    
    if await check_file_hash(db, file_hash):
        raise HTTPException(status_code=409, detail="This document has already been uploaded.")

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(file_content)
            temp_path = temp_file.name
        
        result = await rag_service.process_pdf_file(db, temp_path, safe_filename, file_hash)
        
        if result["status"] == "success":
            return UploadResponse(status="success", filename=result["filename"], pages=result["pages"], chunks=result["chunks"], message=f"Successfully processed '{result['filename']}'.")
        else:
            raise HTTPException(status_code=500, detail=result["error"])
    except Exception as e:
        logger.error(f"Upload processing failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during file processing: {e}")
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)

# Other endpoints remain the same
@router.delete("/documents/{document_id}", response_model=DeleteResponse)
async def delete_document_endpoint(
    document_id: str,
    db: AsyncSession = Depends(get_db),
    rag_service: RAGService = Depends(get_rag_service)
) -> DeleteResponse:
    if not re.match(r'^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$', document_id, re.IGNORECASE):
        raise HTTPException(status_code=422, detail="Invalid document ID format.")
    success = await rag_service.delete_document(db, document_id)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found.")
    return DeleteResponse(status="success", message="Document deleted successfully.")

@router.delete("/documents", response_model=DeleteResponse)
async def clear_all_documents_endpoint(db: AsyncSession = Depends(get_db), rag_service: RAGService = Depends(get_rag_service)):
    await rag_service.clear_all_documents(db)
    return DeleteResponse(status="success", message="All documents cleared successfully.")

@router.delete("/search-history", response_model=DeleteResponse)
async def clear_search_history_endpoint(db: AsyncSession = Depends(get_db)):
    try:
        await db.execute(delete(Message).where(Message.sender == "search"))
        await db.commit()
        logger.info("Search history has been cleared.")
        return DeleteResponse(status="success", message="Search history cleared successfully.")
    except Exception as e:
        logger.error(f"Failed to clear search history: {e}", exc_info=True)
        await db.rollback()
        raise HTTPException(status_code=500, detail="Failed to clear search history.")

@router.get("/documents", response_model=DocumentsListResponse)
async def get_documents_list_endpoint(db: AsyncSession = Depends(get_db), rag_service: RAGService = Depends(get_rag_service)):
    documents_list = await rag_service.list_documents(db)
    return DocumentsListResponse(documents=[DocumentsListItem(**doc) for doc in documents_list])

@router.get("/search-history", response_model=List[Dict])
async def get_search_history_endpoint(db: AsyncSession = Depends(get_db)):
    history = await get_search_history(db, limit=50)
    return history

@router.get("/status", response_model=StatusResponse)
async def get_status_endpoint(db: AsyncSession = Depends(get_db), rag_service: RAGService = Depends(get_rag_service)):
    docs = await rag_service.list_documents(db)
    doc_names = ", ".join([doc['filename'] for doc in docs])
    chunks_count = await rag_service.get_chunks_count()
    return StatusResponse(document_loaded=doc_names if doc_names else None, chunks_available=chunks_count, ready_for_queries=len(docs) > 0)

@router.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_interface():
    try:
        with open("static/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Interface file not found.")