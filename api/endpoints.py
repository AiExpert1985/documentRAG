# api/endpoints.py
import logging
import tempfile
import os
from typing import List

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import HTMLResponse

from services.config import LOGGER_NAME
from services.rag_service import RAGService
from services.llm_service import LLMService
from database.chat_db import get_file_hash, check_file_hash, save_file_hash

from api.types import (
    ChatRequest, ChatResponse, UploadResponse, StatusResponse,
    DocumentsListResponse, DeleteResponse, Message as MessageType
)

# This import must be at the top level to avoid circular imports.
# The actual services are not instantiated here.
import main

logger = logging.getLogger(LOGGER_NAME)

router = APIRouter()

def get_rag_service() -> RAGService:
    """Provides the singleton instance of RAGService."""
    if main._rag_service is None:
        raise HTTPException(status_code=503, detail="RAG service is not initialized.")
    return main._rag_service

def get_llm_service() -> LLMService:
    """Provides the singleton instance of LLMService."""
    if main._llm_service is None:
        raise HTTPException(status_code=503, detail="LLM service is not initialized.")
    return main._llm_service

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    rag_service: RAGService = Depends(get_rag_service),
    llm_service: LLMService = Depends(get_llm_service)
) -> ChatResponse:
    if not rag_service.has_documents():
        raise HTTPException(status_code=400, detail="Please upload a PDF document first.")

    await rag_service.save_chat_message("user", request.question)
    relevant_chunks = await rag_service.retrieve_chunks(request.question)
    
    if not relevant_chunks:
        ai_response = "No relevant content found for your question."
        await rag_service.save_chat_message("ai", ai_response)
        return ChatResponse(status="error", error=ai_response)
        
    context = ""
    sources = set()
    for chunk in relevant_chunks:
        context += f"Source: {chunk['metadata']['document_name']}, Page: {chunk['metadata']['page']}\n{chunk['content']}\n\n"
        sources.add(f"{chunk['metadata']['document_name']}, p. {chunk['metadata']['page']}")
    
    sources_str = ", ".join(sources)

    chat_history = await rag_service.get_chat_history(limit=10)
    history_prompt = "\n".join([f"{msg['sender']}: {msg['content']}" for msg in chat_history[-5:]])
    
    prompt = f"""Based on the following context and chat history, answer the question accurately.

Chat History:
{history_prompt}

Context:
{context}

Question: {request.question}

Answer:"""
    
    result = llm_service.chat(prompt)
    
    if result["status"] == "success":
        await rag_service.save_chat_message("ai", result["answer"])
        return ChatResponse(
            status="success",
            answer=result["answer"],
            document=sources_str,
            chunks_used=len(relevant_chunks)
        )
    else:
        error_msg = f"AI error: {result.get('error', 'Unknown error')}"
        await rag_service.save_chat_message("ai", error_msg)
        # Consistent error return
        return ChatResponse(status="error", error=error_msg)

@router.post("/upload-pdf", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...), rag_service: RAGService = Depends(get_rag_service)) -> UploadResponse:
    if not file.filename or not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    if file.size and file.size > 50_000_000:
        raise HTTPException(status_code=400, detail="File too large (50MB max)")
    
    file_content = await file.read()
    file_hash = get_file_hash(file_content)
    
    if await check_file_hash(file_hash):
        raise HTTPException(status_code=409, detail="Document already uploaded")

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(file_content)
            temp_path = temp_file.name
        
        result = await rag_service.process_pdf_file(temp_path, file.filename)
        
        if result["status"] == "success":
            await save_file_hash(file_hash)
            return UploadResponse(
                status="success",
                filename=result["filename"],
                pages=result["pages"],
                chunks=result["chunks"],
                message=result["message"],
                retrieval_method=result["retrieval_method"]
            )
        else:
            raise HTTPException(status_code=500, detail=result["error"])
            
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail="Upload failed")
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)

@router.delete("/documents/{document_id}", response_model=DeleteResponse)
async def delete_document_endpoint(document_id: str, rag_service: RAGService = Depends(get_rag_service)):
    success = await rag_service.delete_document(document_id)
    if success:
        return DeleteResponse(status="success", message="Document deleted")
    else:
        raise HTTPException(status_code=404, detail="Document not found")

@router.delete("/documents", response_model=DeleteResponse)
async def clear_all_documents_endpoint(rag_service: RAGService = Depends(get_rag_service)):
    await rag_service.clear_all_documents()
    return DeleteResponse(status="success", message="All documents cleared")

@router.get("/documents", response_model=DocumentsListResponse)
def get_documents_list(rag_service: RAGService = Depends(get_rag_service)):
    documents_list = [
        {"id": doc_id, "filename": filename}
        for doc_id, filename in rag_service.loaded_documents.items()
    ]
    return DocumentsListResponse(documents=documents_list)

@router.get("/chat-history", response_model=List[MessageType])
async def get_chat_history_endpoint(rag_service: RAGService = Depends(get_rag_service)):
    try:
        history = await rag_service.get_chat_history(limit=50)
        return [MessageType(sender=msg["sender"], content=msg["content"]) for msg in history]
    except Exception as e:
        logger.error(f"Chat history error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get chat history")

@router.get("/status", response_model=StatusResponse)
def get_status(rag_service: RAGService = Depends(get_rag_service)) -> StatusResponse:
    try:
        status = rag_service.get_status()
        return StatusResponse(
            current_method=status["current_method"],
            document_loaded=status["document_loaded"],
            chunks_available=status["chunks_available"],
            ready_for_queries=status["ready_for_queries"]
        )
    except Exception as e:
        logger.error(f"Status error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get status")

@router.get("/", response_class=HTMLResponse)
async def serve_interface() -> HTMLResponse:
    try:
        with open("static/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail="index.html is missing. Please ensure the path is correct."
        )