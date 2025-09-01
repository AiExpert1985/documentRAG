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

logger = logging.getLogger(LOGGER_NAME)
router = APIRouter()

# Proper singleton using FastAPI's dependency system
_rag_service_instance = None
_llm_service_instance = None

def get_rag_service() -> RAGService:
    global _rag_service_instance
    if _rag_service_instance is None:
        _rag_service_instance = RAGService()
    return _rag_service_instance

def get_llm_service() -> LLMService:
    global _llm_service_instance
    if _llm_service_instance is None:
        _llm_service_instance = LLMService()
    return _llm_service_instance

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    rag_service: RAGService = Depends(get_rag_service),
    llm_service: LLMService = Depends(get_llm_service)
) -> ChatResponse:
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
        sources = set()
        for chunk in relevant_chunks:
            context += f"Source: {chunk['metadata']['document_name']}, Page: {chunk['metadata']['page']}\n{chunk['content']}\n\n"
            sources.add(f"{chunk['metadata']['document_name']}, p. {chunk['metadata']['page']}")
        
        sources_str = ", ".join(sources)

        # Get chat history
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
        logger.error(f"Chat error: {e}")
        return ChatResponse(status="error", error="Processing error occurred")

@router.post("/upload-pdf", response_model=UploadResponse)
async def upload_pdf(
    file: UploadFile = File(...),
    rag_service: RAGService = Depends(get_rag_service)
) -> UploadResponse:
    if not file.filename or not file.filename.endswith('.pdf'):
        return UploadResponse(status="error", error="Only PDF files are supported")
    
    if file.size and file.size > 50_000_000:
        return UploadResponse(status="error", error="File too large (50MB max)")
    
    try:
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
                
        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
                
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return UploadResponse(status="error", error="Upload failed")

@router.delete("/documents/{document_id}", response_model=DeleteResponse)
async def delete_document_endpoint(
    document_id: str,
    rag_service: RAGService = Depends(get_rag_service)
) -> DeleteResponse:
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
async def clear_all_documents_endpoint(
    rag_service: RAGService = Depends(get_rag_service)
) -> DeleteResponse:
    try:
        await rag_service.clear_all_documents()
        return DeleteResponse(status="success", message="All documents cleared")
    except Exception as e:
        logger.error(f"Clear error: {e}")
        return DeleteResponse(status="error", message="Clear failed")

@router.get("/documents", response_model=DocumentsListResponse)
def get_documents_list(
    rag_service: RAGService = Depends(get_rag_service)
) -> DocumentsListResponse:
    try:
        documents_list = [
            {"id": doc_id, "filename": filename}
            for doc_id, filename in rag_service.loaded_documents.items()
        ]
        return DocumentsListResponse(documents=documents_list)
    except Exception as e:
        logger.error(f"Documents list error: {e}")
        return DocumentsListResponse(documents=[])

@router.get("/chat-history", response_model=List[MessageType])
async def get_chat_history_endpoint(
    rag_service: RAGService = Depends(get_rag_service)
) -> List[MessageType]:
    try:
        history = await rag_service.get_chat_history(limit=50)
        return [MessageType(sender=msg["sender"], content=msg["content"]) for msg in history]
    except Exception as e:
        logger.error(f"Chat history error: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve chat history")

@router.get("/status", response_model=StatusResponse)
def get_status(
    rag_service: RAGService = Depends(get_rag_service)
) -> StatusResponse:
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
        with open("static/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Interface file not found"
        )

# Cleanup function for shutdown
async def cleanup_services():
    global _rag_service_instance, _llm_service_instance
    try:
        if _rag_service_instance:
            await _rag_service_instance.clear_all_documents()
        _rag_service_instance = None
        _llm_service_instance = None
    except Exception as e:
        logger.error(f"Cleanup error: {e}")