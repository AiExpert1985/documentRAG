# services/rag_service.py
"""Memory-optimized RAG service for document processing"""

import logging
from typing import List, Optional, Dict, Union
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path

from services.search_strategies import SearchMethod, SearchStrategy, KeywordSearch, SemanticSearch, HybridSearch

# Use the logger configured by logger_config.py
logger = logging.getLogger("rag_system_logger")

class RAGService:
    """Main service for RAG document processing and search"""
    
    def __init__(self, search_method: SearchMethod = SearchMethod.HYBRID) -> None:
        self.current_document: Optional[str] = None
        
        # Strategy objects - created only when needed (lazy loading)
        self._search_strategies: Dict[SearchMethod, SearchStrategy] = {}
        
        # Set search method (defaults to hybrid)
        self.search_method = search_method
        self._current_search_strategy = self._get_strategy(search_method)
    
    def _get_strategy(self, method: SearchMethod) -> SearchStrategy:
        """Get search strategy instance - lazy initialization"""
        if method not in self._search_strategies:
            if method == SearchMethod.SEMANTIC:
                self._search_strategies[method] = SemanticSearch()
            elif method == SearchMethod.KEYWORD:
                self._search_strategies[method] = KeywordSearch()
            elif method == SearchMethod.HYBRID:
                self._search_strategies[method] = HybridSearch()
            else:
                raise ValueError(f"Unsupported search method: {method}")
        
        return self._search_strategies[method]
    
    def get_search_method(self) -> SearchMethod:
        """Get current search method"""
        return self.search_method
    
    def set_search_method(self, method: SearchMethod) -> None:
        """Change search method"""
        self.search_method = method
        self._current_search_strategy = self._get_strategy(method)
        logger.info(f"Search method changed to: {method.value}")
    
    def has_documents(self) -> bool:
        """Check if any documents are loaded"""
        return self._current_search_strategy.get_chunks_count() > 0
    
    def get_chunks_count(self) -> int:
        """Get number of processed chunks"""
        return self._current_search_strategy.get_chunks_count()
    
    def process_pdf_file(self, file_path: str) -> Dict[str, Union[str, int]]:
        """Process PDF file into searchable chunks - memory optimized"""
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
            
            # Setup document store (no longer storing chunks in memory)
            logger.info(f"Processing {len(chunks)} chunks using {self.search_method.value} search")
            
            # Pass document name to the setup method for persistence
            document_name = Path(file_path).name
            setup_success = self._current_search_strategy.setup_document_store(chunks, document_name)
            
            if not setup_success:
                logger.error("Failed to setup document store")
                return {"error": "Failed to setup document storage", "status": "error"}
            
            # Only store document metadata, not the chunks themselves
            self.current_document = document_name
            
            return {
                "status": "success",
                "filename": self.current_document,
                "pages": len(documents),
                "chunks": len(chunks),
                "message": f"PDF processed into {len(chunks)} searchable chunks using {self.search_method.value} search",
                "search_method": self.search_method.value
            }
            
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            return {"error": str(e), "status": "error"}
    
    def search_chunks(self, question: str, top_k: int = 3) -> List[str]:
        """Search for relevant chunks using current strategy"""
        if not self.has_documents():
            logger.warning("No documents available for search")
            return []
        
        try:
            return self._current_search_strategy.search(question, top_k)
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise
    
    def clear_documents(self) -> Dict[str, Union[str, bool]]:
        """Clear all processed documents"""
        try:
            success = self._current_search_strategy.clear_documents()
            if success:
                self.current_document = None
                logger.info("All documents cleared successfully")
                return {"status": "success", "message": "All documents cleared"}
            else:
                return {"status": "error", "error": "Failed to clear documents"}
        except Exception as e:
            logger.error(f"Error clearing documents: {e}")
            return {"status": "error", "error": str(e)}
    
    def get_status(self) -> Dict[str, Union[str, int, bool]]:
        """Get detailed system status"""
        return {
            "current_method": self.search_method.value,
            "document_loaded": self.current_document,
            "chunks_available": self.get_chunks_count(),
            "ready_for_queries": self.has_documents()
        }