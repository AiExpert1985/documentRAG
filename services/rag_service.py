# services/rag_service.py
"""Memory-optimized RAG service for document processing"""

import logging
from typing import List, Optional, Dict, Union
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path
import uuid
from sqlalchemy.orm import Session

from services.retrieval_strategies import RetrievalMethod, RetrievalStrategy, KeywordRetrieval, SemanticRetrieval, HybridRetrieval
from services.config import LOGGER_NAME
from database.chat_db import get_db, Message

logger = logging.getLogger(LOGGER_NAME)

class RAGService:
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, retrieval_method: RetrievalMethod = RetrievalMethod.SEMANTIC):
        if not hasattr(self, '_initialized'):
            self._current_strategy: RetrievalStrategy = self._create_strategy(retrieval_method)
            self.retrieval_method = retrieval_method
            self.loaded_documents: Dict[str, str] = {}
            self._initialized = True
            logger.info("RAGService initialized")
            
    def _create_strategy(self, method: RetrievalMethod) -> RetrievalStrategy:
        if method == RetrievalMethod.SEMANTIC:
            return SemanticRetrieval()
        elif method == RetrievalMethod.KEYWORD:
            return KeywordRetrieval()
        elif method == RetrievalMethod.HYBRID:
            return HybridRetrieval()

    def set_retrieval_method(self, method: RetrievalMethod):
        if self.retrieval_method == method:
            return
        
        self._current_strategy = self._create_strategy(method)
        self.retrieval_method = method
        logger.info(f"Switched retrieval method to {method.value}")
    
    def has_documents(self) -> bool:
        return len(self.loaded_documents) > 0
    
    def get_chunks_count(self) -> int:
        return self._current_strategy.get_chunks_count()
    
    def process_pdf_file(self, file_path: str) -> Dict[str, Union[str, int]]:
        document_id = str(uuid.uuid4())
        document_name = Path(file_path).name
        
        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            if not documents:
                return {"error": "No content extracted from PDF", "status": "error"}
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = text_splitter.split_documents(documents)
            
            if not chunks:
                return {"error": "Failed to create chunks from document", "status": "error"}
            
            logger.info(f"Processing {len(chunks)} chunks using {self.retrieval_method.value} retrieval")
            setup_success = self._current_strategy.add_document(document_id, document_name, chunks)
            
            if not setup_success:
                logger.error("Failed to setup document store")
                return {"error": "Failed to setup document storage", "status": "error"}
            
            self.loaded_documents[document_id] = document_name
            
            return {
                "status": "success",
                "filename": document_name,
                "pages": len(documents),
                "chunks": len(chunks),
                "message": f"PDF processed into {len(chunks)} retrievalable chunks using {self.retrieval_method.value} retrieval",
                "retrieval_method": self.retrieval_method.value
            }
            
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            return {"error": str(e), "status": "error"}
    
    def retrieval_chunks(self, question: str, top_k: int = 3) -> List[Dict]:
        if not self.has_documents():
            logger.warning("No documents available for retrieval")
            return []
        
        try:
            return self._current_strategy.retrieval(question, top_k)
        except Exception as e:
            logger.error(f"retrieval failed: {e}")
            raise
    
    def get_status(self) -> Dict[str, Union[str, int, bool]]:
        return {
            "current_method": self.retrieval_method.value,
            "document_loaded": ", ".join(self.loaded_documents.values()) if self.loaded_documents else None,
            "chunks_available": self.get_chunks_count(),
            "ready_for_queries": self.has_documents()
        }
        
    def save_chat_message(self, sender: str, content: str):
        db_gen = get_db()
        db = next(db_gen)
        try:
            new_message = Message(sender=sender, content=content)
            db.add(new_message)
            db.commit()
            db.refresh(new_message)
        except Exception as e:
            logger.error(f"Failed to save chat message: {e}")
        finally:
            db.close()
            
    def get_chat_history(self) -> List[Dict]:
        db_gen = get_db()
        db = next(db_gen)
        try:
            messages = db.query(Message).order_by(Message.timestamp).all()
            return [{"sender": msg.sender, "content": msg.content} for msg in messages]
        except Exception as e:
            logger.error(f"Failed to retrieve chat history: {e}")
            return []
        finally:
            db.close()
            
    def delete_document(self, document_id: str) -> bool:
        if document_id not in self.loaded_documents:
            return False
        
        success = self._current_strategy.delete_document(document_id)
        if success:
            del self.loaded_documents[document_id]
        return success

    def clear_all_documents(self) -> bool:
        success = self._current_strategy.clear_all_documents()
        if success:
            self.loaded_documents.clear()
        return success