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

from services.config import settings
from services.rag_service import RAGService
from services.llm_service import LLMService
from database.chat_db import get_file_hash, check_file_hash, AsyncSessionLocal, Message, Document
from api.types import ChatRequest, ChatResponse, UploadResponse, StatusResponse, DocumentsListResponse, DeleteResponse, Message as MessageType, DocumentsListItem

logger = logging.getLogger(settings.LOGGER_NAME)
router = APIRouter()

# Dependency Injection and Chat History Utilities remain the same...
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session
def get_rag_service(request: Request) -> RAGService:
    return request.app.state.rag_service
def get_llm_service(request: Request) -> LLMService:
    return request.app.state.llm_service
async def save_chat_message(db: AsyncSession, sender: str, content: str):
    try:
        new_message = Message(sender=sender, content=content)
        db.add(new_message)
        await db.commit()
    except Exception as e:
        logger.error(f"Failed to save chat message: {e}")
        await db.rollback()
async def get_chat_history(db: AsyncSession, limit: int = 10) -> List[Dict]:
    try:
        result = await db.execute(select(Message).order_by(Message.timestamp.desc()).limit(limit))
        messages = result.scalars().all()
        return [{"sender": msg.sender, "content": msg.content} for msg in reversed(messages)]
    except Exception as e:
        logger.error(f"Failed to get chat history: {e}")
        return []

# --- Updated API Endpoints ---
@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
    rag_service: RAGService = Depends(get_rag_service),
    llm_service: LLMService = Depends(get_llm_service)
) -> ChatResponse:
    if not 3 <= len(request.question) <= 2000:
        raise HTTPException(status_code=422, detail="Question must be between 3 and 2000 characters.")
    if not await rag_service.has_documents(db):
        raise HTTPException(status_code=400, detail="Please upload a PDF document first.")
    
    # The rest of the function logic is correct and remains the same...
    try:
        await save_chat_message(db, "user", request.question)
        relevant_chunks = await rag_service.retrieve_chunks(db, request.question)
        if not relevant_chunks:
            ai_response = "I could not find any relevant content in the document to answer your question."
            await save_chat_message(db, "ai", ai_response)
            return ChatResponse(status="success", answer=ai_response)
        context = ""
        sources = set()
        for chunk in relevant_chunks:
            context += f"Source: {chunk['metadata']['document_name']}, Page: {chunk['metadata']['page']}\n{chunk['content']}\n\n"
            sources.add(f"{chunk['metadata']['document_name']}, p. {chunk['metadata']['page']}")
        sources_str = ", ".join(sorted(list(sources)))
        chat_history = await get_chat_history(db, limit=settings.CHAT_CONTEXT_LIMIT)
        history_prompt = "\n".join([f"{msg['sender']}: {msg['content']}" for msg in chat_history])
        prompt = f"""Based on the provided context and chat history, give a precise and helpful answer to the user's question.
                Respond in the same language as the user's question.
                Cite any sources used from the context.

                Chat History:
                {history_prompt}

                Context:
                {context}

                Question: {request.question}

                Answer:"""
        result = llm_service.chat(prompt)
        if result["status"] == "success":
            await save_chat_message(db, "ai", result["answer"])
            return ChatResponse(status="success", answer=result["answer"], document=sources_str, chunks_used=len(relevant_chunks))
        else:
            error_msg = f"AI service error: {result.get('error', 'Unknown error')}"
            await save_chat_message(db, "ai", error_msg)
            raise HTTPException(status_code=503, detail=error_msg)
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal processing error occurred.")

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

# The other endpoints (clear_all, clear_history, get_documents, etc.) are correct and remain the same...
@router.delete("/documents", response_model=DeleteResponse)
async def clear_all_documents_endpoint(db: AsyncSession = Depends(get_db), rag_service: RAGService = Depends(get_rag_service)):
    await rag_service.clear_all_documents(db)
    return DeleteResponse(status="success", message="All documents cleared successfully.")

@router.delete("/chat-history", response_model=DeleteResponse)
async def clear_chat_history_endpoint(db: AsyncSession = Depends(get_db)):
    try:
        await db.execute(delete(Message))
        await db.commit()
        logger.info("Chat history has been cleared.")
        return DeleteResponse(status="success", message="Chat history cleared successfully.")
    except Exception as e:
        logger.error(f"Failed to clear chat history: {e}", exc_info=True)
        await db.rollback()
        raise HTTPException(status_code=500, detail="Failed to clear chat history.")

@router.get("/documents", response_model=DocumentsListResponse)
async def get_documents_list_endpoint(db: AsyncSession = Depends(get_db), rag_service: RAGService = Depends(get_rag_service)):
    documents_list = await rag_service.list_documents(db)
    return DocumentsListResponse(documents=[DocumentsListItem(**doc) for doc in documents_list])

@router.get("/chat-history", response_model=List[MessageType])
async def get_chat_history_endpoint(db: AsyncSession = Depends(get_db)):
    history = await get_chat_history(db, limit=100)
    return [MessageType(sender=msg["sender"], content=msg["content"]) for msg in history]

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