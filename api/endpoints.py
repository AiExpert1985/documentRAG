# api/endpoints.py
import logging
import tempfile
import os
from typing import List

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse

from services.config import LOGGER_NAME
from services.rag_service import RAGService
from services.llm_service import LLMService
from database.chat_db import get_file_hash, check_file_hash, save_file_hash

from api.types import (
    ChatRequest, ChatResponse, UploadResponse, StatusResponse,
    DocumentsListResponse, DeleteResponse, Message as MessageType
)

logger = logging.getLogger(LOGGER_NAME)

router = APIRouter()
rag_service = RAGService()
llm_service = LLMService()

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    if not rag_service.has_documents():
        return ChatResponse(
            status="no_document",
            error="Please upload a PDF document first."
        )
    
    try:
        await rag_service.save_chat_message("user", request.question)
        relevant_chunks = await rag_service.retrieve_chunks(request.question)
        
        if not relevant_chunks:
            ai_response = "No relevant content found for your question."
            await rag_service.save_chat_message("ai", ai_response)
            return ChatResponse(status="error", error=ai_response)
            
        # Build context
        context = ""
        sources = []
        for chunk in relevant_chunks:
            context += f"Source: {chunk['metadata']['document_name']}, Page: {chunk['metadata']['page']}\n{chunk['content']}\n\n"
            sources.append(f"{chunk['metadata']['document_name']}, p. {chunk['metadata']['page']}")
        
        sources_str = ", ".join(sources)

        # Get chat history for context
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
            return ChatResponse(status="error", error=error_msg)
            
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        return ChatResponse(status="error", error="Processing error occurred")

@router.post("/upload-pdf", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)) -> UploadResponse:
    if not file.filename or not file.filename.endswith('.pdf'):
        return UploadResponse(status="error", error="Only PDF files are supported")
    
    if file.size and file.size > 50_000_000:  # 50MB limit
        return UploadResponse(status="error", error="File too large (50MB max)")
    
    # Check for duplicates
    file_content = await file.read()
    file_hash = get_file_hash(file_content)
    
    if await check_file_hash(file_hash):
        return UploadResponse(status="error", error="Document already uploaded")

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
            return UploadResponse(status="error", error=result["error"])
            
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return UploadResponse(status="error", error="Upload failed")
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)

@router.delete("/documents/{document_id}", response_model=DeleteResponse)
async def delete_document_endpoint(document_id: str):
    try:
        success = await rag_service.delete_document(document_id)
        if success:
            return DeleteResponse(status="success", message="Document deleted")
        else:
            return DeleteResponse(status="error", message="Document not found")
    except Exception as e:
        logger.error(f"Delete error: {e}")
        return DeleteResponse(status="error", message="Delete failed")

@router.delete("/documents", response_model=DeleteResponse)
async def clear_all_documents_endpoint():
    try:
        await rag_service.clear_all_documents()
        return DeleteResponse(status="success", message="All documents cleared")
    except Exception as e:
        logger.error(f"Clear error: {e}")
        return DeleteResponse(status="error", message="Clear failed")

@router.get("/documents", response_model=DocumentsListResponse)
def get_documents_list():
    documents_list = [
        {"id": doc_id, "filename": filename}
        for doc_id, filename in rag_service.loaded_documents.items()
    ]
    return DocumentsListResponse(documents=documents_list)

@router.get("/chat-history", response_model=List[MessageType])
async def get_chat_history_endpoint():
    try:
        history = await rag_service.get_chat_history(limit=50)
        return [MessageType(sender=msg["sender"], content=msg["content"]) for msg in history]
    except Exception as e:
        logger.error(f"Chat history error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get chat history")

@router.get("/status", response_model=StatusResponse)
def get_status() -> StatusResponse:
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
        return StatusResponse(
            current_method="error",
            document_loaded=None,
            chunks_available=0,
            ready_for_queries=False
        )

@router.get("/", response_class=HTMLResponse)
async def serve_interface() -> HTMLResponse:
    try:
        with open("templates/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>Interface not found</h1><p>templates/index.html is missing</p>",
            status_code=500
        )