# services/rag_service.py

from typing import List, Optional, Any, Dict, Union
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path

class RAGService:
   def __init__(self) -> None:
       self.processed_chunks: List[Any] = []
       self.current_document: Optional[str] = None
   
   def process_pdf_file(self, file_path: str) -> Dict[str, Union[str, int]]:
       try:
           loader: PyPDFLoader = PyPDFLoader(file_path)
           documents: List[Any] = loader.load()
           
           text_splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(
               chunk_size=1000,
               chunk_overlap=200
           )
           chunks: List[Any] = text_splitter.split_documents(documents)
           
           self.processed_chunks = chunks
           self.current_document = Path(file_path).name
           
           return {
               "status": "success",
               "filename": self.current_document,
               "pages": len(documents),
               "chunks": len(chunks),
               "message": f"PDF processed into {len(chunks)} searchable chunks"
           }
       except Exception as e:
           return {"error": str(e), "status": "error"}
   
   def search_chunks(self, question: str) -> List[str]:
       if not self.processed_chunks:
           return []
       
       relevant_chunks: List[str] = []
       question_lower: str = question.lower()
       
       for chunk in self.processed_chunks:
           if any(word in chunk.page_content.lower() for word in question_lower.split()):
               relevant_chunks.append(chunk.page_content)
           if len(relevant_chunks) >= 3:
               break
       
       if not relevant_chunks:
           relevant_chunks = [self.processed_chunks[0].page_content]
       
       return relevant_chunks