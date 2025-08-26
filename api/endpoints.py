# api/endpoints.py

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
import tempfile
import os
from services.rag_service import RAGService
from services.llm_service import LLMService
from api.models import ChatRequest, QueryRequest, LocalPDFRequest

router: APIRouter = APIRouter()
rag_service: RAGService = RAGService()
llm_service: LLMService = LLMService()

@router.post("/chat")
@router.post("/chat")
def chat_endpoint(request: ChatRequest):
    # Check if user is trying to ask about documents without uploading
    if not rag_service.processed_chunks and any(word in request.prompt.lower() for word in ['document', 'pdf', 'file', 'text', 'content']):
        return {
            "answer": "You didn't upload any PDF. Please upload a document first before asking questions about it.",
            "status": "success"
        }
    
    # For general chat without PDF context
    return llm_service.chat(request.prompt)

@router.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename or not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
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

@router.post("/query-pdf")
def query_pdf(request: QueryRequest):
    if not rag_service.processed_chunks:
        return {"error": "No PDF loaded. Upload a PDF first.", "status": "error"}
    
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

@router.get("/status")
def get_status():
    return {
        "document_loaded": rag_service.current_document,
        "chunks_available": len(rag_service.processed_chunks),
        "ready_for_queries": len(rag_service.processed_chunks) > 0
    }

@router.get("/", response_class=HTMLResponse)
async def serve_interface():
    with open("templates/index.html", "r") as f:
        return HTMLResponse(content=f.read())