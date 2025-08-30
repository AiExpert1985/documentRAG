# services/rag_service.py
"""Main RAG service for document processing and search orchestration"""

from typing import List, Optional, Any, Dict, Union
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path

from .search_strategies import (
    SearchStrategy, 
    SearchMethod, 
    KeywordSearch, 
    SemanticSearch
)

class RAGService:
    """Main service for RAG document processing and search"""
    
    def __init__(self, search_method: SearchMethod = SearchMethod.SEMANTIC) -> None:
        self.processed_chunks: List[Any] = []
        self.current_document: Optional[str] = None
        
        # Initialize search strategies
        self.keyword_search = KeywordSearch()
        self.semantic_search = SemanticSearch()
        
        # Set default search method
        self.search_method = search_method
        self._current_search_strategy = self._get_search_strategy(search_method)
    
    def _get_search_strategy(self, method: SearchMethod) -> SearchStrategy:
        """Get the appropriate search strategy"""
        if method == SearchMethod.SEMANTIC and self.semantic_search.is_available():
            return self.semantic_search
        else:
            # Fallback to keyword search
            return self.keyword_search
    
    def set_search_method(self, method: SearchMethod) -> bool:
        """Change search method and return success status"""
        strategy = self._get_search_strategy(method)
        
        if method == SearchMethod.SEMANTIC and not strategy.is_available():
            print("Semantic search not available, using keyword search")
            return False
        
        self.search_method = method
        self._current_search_strategy = strategy
        
        # If switching to semantic and we have chunks, setup vector store
        if (method == SearchMethod.SEMANTIC and 
            self.processed_chunks and 
            isinstance(strategy, SemanticSearch)):
            return strategy.setup_vector_store(self.processed_chunks)
        
        return True
    
    def get_available_search_methods(self) -> List[SearchMethod]:
        """Get list of available search methods"""
        methods = [SearchMethod.KEYWORD]  # Always available
        
        if self.semantic_search.is_available():
            methods.append(SearchMethod.SEMANTIC)
        
        return methods
    
    def has_documents(self) -> bool:
        """Check if any documents are loaded and processed"""
        return len(self.processed_chunks) > 0
    
    def process_pdf_file(self, file_path: str) -> Dict[str, Union[str, int]]:
        """Process PDF file into searchable chunks"""
        try:
            # Load PDF document
            loader: PyPDFLoader = PyPDFLoader(file_path)
            documents: List[Any] = loader.load()
            
            if not documents:
                return {"error": "No content extracted from PDF", "status": "error"}
            
            # Split into chunks
            text_splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks: List[Any] = text_splitter.split_documents(documents)
            
            if not chunks:
                return {"error": "Failed to create chunks from document", "status": "error"}
            
            # Store chunks
            self.processed_chunks = chunks
            self.current_document = Path(file_path).name
            
            # Setup vector store if using semantic search
            if (self.search_method == SearchMethod.SEMANTIC and 
                isinstance(self._current_search_strategy, SemanticSearch)):
                vector_setup_success = self._current_search_strategy.setup_vector_store(chunks)
                if not vector_setup_success:
                    print("Warning: Failed to setup vector store, falling back to keyword search")
                    self.set_search_method(SearchMethod.KEYWORD)
            
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
        """Search for relevant chunks using the current search strategy"""
        if not self.processed_chunks:
            return []
        
        return self._current_search_strategy.search(question, self.processed_chunks, top_k)
    
    def get_search_info(self) -> Dict[str, Any]:
        """Get information about current search configuration"""
        return {
            "current_method": self.search_method.value,
            "available_methods": [method.value for method in self.get_available_search_methods()],
            "semantic_available": self.semantic_search.is_available(),
            "chunks_loaded": len(self.processed_chunks)
        }