# api/endpoints.py
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional
import tempfile
import os
from services.rag_service import RAGService
from services.llm_service import LLMService

router: APIRouter = APIRouter()
rag_service: RAGService = RAGService()
llm_service: LLMService = LLMService()

# Pydantic Response Models
class ChatResponse(BaseModel):
    status: str
    answer: Optional[str] = None
    error: Optional[str] = None
    document: Optional[str] = None
    chunks_used: Optional[int] = None

class UploadResponse(BaseModel):
    status: str
    filename: Optional[str] = None
    pages: Optional[int] = None
    chunks: Optional[int] = None
    message: Optional[str] = None
    error: Optional[str] = None

class StatusResponse(BaseModel):
    document_loaded: Optional[str] = None
    chunks_available: int = 0
    ready_for_queries: bool = False

# Request Models
class ChatRequest(BaseModel):
    question: str

class LocalPDFRequest(BaseModel):
    pdf_path: str

@router.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """Single chat endpoint - requires PDF to be uploaded first"""
    
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
                message=result["message"]
            )
        else:
            return UploadResponse(
                status="error",
                error=result["error"]
            )
            
    except Exception as e:
        return UploadResponse(
            status="error",
            error=f"Upload failed: {str(e)}"
        )
    finally:
        # Cleanup temporary file
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except:
                pass

@router.post("/process-local-pdf", response_model=UploadResponse)
def process_local_pdf(request: LocalPDFRequest) -> UploadResponse:
    """Process PDF file from local file system"""
    
    if not os.path.exists(request.pdf_path):
        return UploadResponse(
            status="error",
            error="File not found"
        )
    
    try:
        result = rag_service.process_pdf_file(request.pdf_path)
        
        if result["status"] == "success":
            return UploadResponse(
                status="success",
                filename=result["filename"],
                pages=result["pages"],
                chunks=result["chunks"],
                message=result["message"]
            )
        else:
            return UploadResponse(
                status="error",
                error=result["error"]
            )
            
    except Exception as e:
        return UploadResponse(
            status="error",
            error=f"Processing failed: {str(e)}"
        )

@router.get("/status", response_model=StatusResponse)
def get_status() -> StatusResponse:
    """Get current system status"""
    return StatusResponse(
        document_loaded=rag_service.current_document,
        chunks_available=len(rag_service.processed_chunks),
        ready_for_queries=rag_service.has_documents()
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