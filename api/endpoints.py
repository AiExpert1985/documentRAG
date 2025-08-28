# api/endpoints.py
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from typing import Dict, Union, Any
import tempfile
import os
from services.rag_service import RAGService
from services.llm_service import LLMService
from api.models import ChatRequest, QueryRequest, LocalPDFRequest

router: APIRouter = APIRouter()
rag_service: RAGService = RAGService()
llm_service: LLMService = LLMService()

@router.post("/chat")
def chat_endpoint(request: ChatRequest) -> Dict[str, Union[str, Any]]:
    """General chat with LLM (no PDF context)"""
    return llm_service.chat(request.prompt)

@router.post("/chat-pdf")
def chat_with_pdf(request: QueryRequest) -> Dict[str, Union[str, Any]]:
    """Smart chat with PDF context or friendly message if no PDF"""
    if not rag_service.processed_chunks:
        return {
            "answer": "You haven't uploaded any PDF yet. Please upload a document first to chat with it, or I can help you with general questions!",
            "status": "success"
        }
    
    relevant_chunks = rag_service.search_chunks(request.question)
    context = "\n\n".join(relevant_chunks)
    
    prompt = f"""Based on the following context from the document "{rag_service.current_document}", answer the question.

Context:
{context}

Question: {request.question}

Answer:"""
    
    result = llm_service.chat(prompt)
    if result["status"] == "success":
        result.update({
            "document": rag_service.current_document,
            "chunks_used": len(relevant_chunks)
        })
    
    return result

@router.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)) -> Dict[str, Union[str, int]]:
    """Upload and process PDF file"""
    if not file.filename or not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            content: bytes = await file.read()
            temp_file.write(content)
            temp_path: str = temp_file.name
        
        result = rag_service.process_pdf_file(temp_path)
        os.unlink(temp_path)
        return result
    except Exception as e:
        if 'temp_path' in locals():
            try:
                os.unlink(temp_path)
            except:
                pass
        return {"error": str(e), "status": "error"}

@router.post("/process-local-pdf")
def process_local_pdf(request: LocalPDFRequest) -> Dict[str, Union[str, int]]:
    """Process PDF file from local file system"""
    return rag_service.process_pdf_file(request.pdf_path)

@router.get("/status")
def get_status() -> Dict[str, Union[str, int, bool]]:
    """Get current system status"""
    return {
        "document_loaded": rag_service.current_document,
        "chunks_available": len(rag_service.processed_chunks),
        "ready_for_queries": len(rag_service.processed_chunks) > 0
    }

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