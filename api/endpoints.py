# api/endpoints.py
import logging
import tempfile
import os
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse

from services.rag_service import RAGService
from services.llm_service import LLMService
from services.search_strategies import SearchMethod
from api.types import (
    ChatRequest, 
    ChatResponse, 
    UploadResponse, 
    StatusResponse,
    SearchMethodRequest,
    ClearDocumentsResponse
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()
rag_service = RAGService()
llm_service = LLMService()

@router.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """Chat with loaded documents using current search strategy"""
    
    # Check if documents are loaded
    if not rag_service.has_documents():
        return ChatResponse(
            status="no_document",
            error="Please upload a PDF document first. I'm designed to answer questions about uploaded documents."
        )
    
    try:
        # Search for relevant chunks
        relevant_chunks = rag_service.search_chunks(request.question)
        
        if not relevant_chunks:
            return ChatResponse(
                status="error",
                error="No relevant content found in the document for your question."
            )
        
        # Build context from chunks
        context = "\n\n".join(relevant_chunks)
        
        # Create prompt with context
        prompt = f"""Based on the following context from the document "{rag_service.current_document}", answer the question accurately and concisely.

                Context:
                {context}

                Question: {request.question}

                Answer:"""
        
        # Get LLM response
        result = llm_service.chat(prompt)
        
        if result["status"] == "success":
            return ChatResponse(
                status="success",
                answer=result["answer"],
                document=rag_service.current_document,
                chunks_used=len(relevant_chunks)
            )
        else:
            return ChatResponse(
                status="error",
                error=f"LLM error: {result.get('error', 'Unknown error')}"
            )
            
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        return ChatResponse(
            status="error",
            error=f"Processing error: {str(e)}"
        )

@router.post("/upload-pdf", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)) -> UploadResponse:
    """Upload and process PDF file"""
    
    # Validation
    if not file.filename or not file.filename.endswith('.pdf'):
        return UploadResponse(
            status="error",
            error="Only PDF files are supported"
        )
    
    if file.size and file.size > 50_000_000:  # 50MB limit
        return UploadResponse(
            status="error",
            error="File too large. Maximum size is 50MB"
        )
    
    temp_path = None
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            content: bytes = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        # Process PDF
        result = rag_service.process_pdf_file(temp_path)
        
        if result["status"] == "success":
            return UploadResponse(
                status="success",
                filename=result["filename"],
                pages=result["pages"],
                chunks=result["chunks"],
                message=result["message"],
                search_method=result["search_method"]
            )
        else:
            return UploadResponse(
                status="error",
                error=result["error"]
            )
            
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return UploadResponse(
            status="error",
            error=f"Upload failed: {str(e)}"
        )
    finally:
        # Cleanup temporary file
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception as e:
                logger.error(f"Failed to cleanup temp file: {e}")

@router.post("/set-search-method")
def set_search_method(request: SearchMethodRequest) -> dict:
    """Change the active search strategy"""
    try:
        # Validate search method
        try:
            search_method = SearchMethod(request.search_method)
        except ValueError:
            available_methods = [method.value for method in SearchMethod]
            return {
                "status": "error",
                "error": f"Invalid search method. Available methods: {available_methods}"
            }
        
        # Set search method
        rag_service.set_search_method(search_method)
        
        return {
            "status": "success",
            "message": f"Search method changed to: {search_method.value}",
            "current_method": search_method.value
        }
        
    except Exception as e:
        logger.error(f"Error setting search method: {e}")
        return {
            "status": "error",
            "error": str(e)
        }

@router.delete("/clear-documents", response_model=ClearDocumentsResponse)
def clear_documents() -> ClearDocumentsResponse:
    """Clear all loaded documents"""
    try:
        result = rag_service.clear_documents()
        return ClearDocumentsResponse(
            status=result["status"],
            message=result.get("message"),
            error=result.get("error")
        )
    except Exception as e:
        logger.error(f"Error clearing documents: {e}")
        return ClearDocumentsResponse(
            status="error",
            error=str(e)
        )

@router.get("/status", response_model=StatusResponse)
def get_status() -> StatusResponse:
    """Get current system status"""
    try:
        status = rag_service.get_status()
        return StatusResponse(
            current_method=status["current_method"],
            document_loaded=status["document_loaded"],
            chunks_available=status["chunks_available"],
            ready_for_queries=status["ready_for_queries"]
        )
    except Exception as e:
        logger.error(f"Status endpoint error: {e}")
        return StatusResponse(
            current_method="unknown",
            document_loaded=None,
            chunks_available=0,
            ready_for_queries=False
        )

@router.get("/", response_class=HTMLResponse)
async def serve_interface() -> HTMLResponse:
    """Serve the web interface"""
    try:
        with open("templates/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="""
        <html><body>
        <h1>Error: templates/index.html not found</h1>
        <p>Please ensure the templates directory exists with index.html</p>
        </body></html>
        """, status_code=500)