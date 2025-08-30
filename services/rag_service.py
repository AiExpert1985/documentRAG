# services/rag_service.py - Simple version

from typing import List, Optional, Any, Dict, Union
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path

from services.search_strategies import SearchMethod, KeywordSearch, SemanticSearch

class RAGService:
    def __init__(self, search_method: SearchMethod = SearchMethod.SEMANTIC) -> None:
        self.processed_chunks: List[Any] = []
        self.current_document: Optional[str] = None
        
        # Initialize search strategies
        self.keyword_search = KeywordSearch()
        self.semantic_search = SemanticSearch()
        
        # Set search method (defaults to semantic)
        self.search_method = search_method
        self._current_search_strategy = self._get_strategy(search_method)
    
    def _get_strategy(self, method: SearchMethod):
        """Get search strategy instance"""
        if method == SearchMethod.SEMANTIC:
            return self.semantic_search
        else:
            return self.keyword_search
    
    def set_search_method(self, method: SearchMethod) -> None:
        """Change search method"""
        self.search_method = method
        self._current_search_strategy = self._get_strategy(method)
    
    def has_documents(self) -> bool:
        return len(self.processed_chunks) > 0
    
    def process_pdf_file(self, file_path: str) -> Dict[str, Union[str, int]]:
        try:
            # Load PDF document
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            if not documents:
                return {"error": "No content extracted from PDF", "status": "error"}
            
            # Split into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = text_splitter.split_documents(documents)
            
            if not chunks:
                return {"error": "Failed to create chunks from document", "status": "error"}
            
            # Store chunks
            self.processed_chunks = chunks
            self.current_document = Path(file_path).name
            
            # Setup vector store if using semantic search
            if self.search_method == SearchMethod.SEMANTIC:
                self.semantic_search.setup_vector_store(chunks)
            
            return {
                "status": "success",
                "filename": self.current_document,
                "pages": len(documents),
                "chunks": len(chunks),
                "message": f"PDF processed into {len(chunks)} searchable chunks using {self.search_method.value} search"
            }
            
        except Exception as e:
            return {"error": str(e), "status": "error"}
    
    def search_chunks(self, question: str, top_k: int = 3) -> List[str]:
        if not self.processed_chunks:
            return []
        
        return self._current_search_strategy.search(question, self.processed_chunks, top_k)