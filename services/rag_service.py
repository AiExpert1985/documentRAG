# services/rag_service.py
import logging
from typing import List, Optional, Dict, Union, Any
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import uuid

from sqlalchemy import select, delete

from services.retrieval_strategies import RetrievalMethod, RetrievalStrategy, KeywordRetrieval, SemanticRetrieval, HybridRetrieval
from services.config import LOGGER_NAME
from database.chat_db import AsyncSessionLocal, Message

logger = logging.getLogger(LOGGER_NAME)

class RAGService:
    def __init__(self, retrieval_method: RetrievalMethod = RetrievalMethod.SEMANTIC):
        self._current_strategy: RetrievalStrategy = self._create_strategy(retrieval_method)
        self.retrieval_method = retrieval_method
        self.loaded_documents: Dict[str, str] = {}
        logger.info("RAG service initialized")

    def _create_strategy(self, method: RetrievalMethod) -> RetrievalStrategy:
        if method == RetrievalMethod.SEMANTIC:
            return SemanticRetrieval()
        elif method == RetrievalMethod.KEYWORD:
            return KeywordRetrieval()
        elif method == RetrievalMethod.HYBRID:
            return HybridRetrieval()

    def has_documents(self) -> bool:
        return len(self.loaded_documents) > 0

    def get_chunks_count(self) -> int:
        return self._current_strategy.get_chunks_count()

    async def process_pdf_file(self, file_path: str, original_filename: str) -> Dict[str, Any]:
        document_id = str(uuid.uuid4())
        
        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            if not documents:
                return {"error": "No content extracted from PDF", "status": "error"}
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(documents)
            
            if not chunks:
                return {"error": "Failed to create chunks", "status": "error"}
            
            success = await self._current_strategy.add_document(document_id, original_filename, chunks)
            
            if not success:
                return {"error": "Failed to store document", "status": "error"}
            
            self.loaded_documents[document_id] = original_filename
            
            return {
                "status": "success",
                "filename": original_filename,
                "pages": len(documents),
                "chunks": len(chunks),
                "message": f"PDF processed into {len(chunks)} chunks",
                "retrieval_method": self.retrieval_method.value
            }
            
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            return {"error": str(e), "status": "error"}

    async def retrieve_chunks(self, question: str, top_k: int = 3) -> List[Dict]:
        if not self.has_documents():
            return []
        
        try:
            return await self._current_strategy.retrieve(question, top_k)
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return []

    def get_status(self) -> Dict[str, Union[str, int, bool]]:
        return {
            "current_method": self.retrieval_method.value,
            "document_loaded": ", ".join(self.loaded_documents.values()) if self.loaded_documents else None,
            "chunks_available": self.get_chunks_count(),
            "ready_for_queries": self.has_documents()
        }
        
    async def save_chat_message(self, sender: str, content: str) -> bool:
        async with AsyncSessionLocal() as db:
            try:
                new_message = Message(sender=sender, content=content)
                db.add(new_message)
                await db.commit()
                return True
            except Exception as e:
                logger.error(f"Failed to save chat message: {e}")
                await db.rollback()
                return False
            
    async def get_chat_history(self, limit: int = 50) -> List[Dict]:
        async with AsyncSessionLocal() as db:
            try:
                result = await db.execute(
                    select(Message).order_by(Message.timestamp.desc()).limit(limit)
                )
                messages = result.scalars().all()
                return [{"sender": msg.sender, "content": msg.content} for msg in reversed(messages)]
            except Exception as e:
                logger.error(f"Failed to get chat history: {e}")
                return []
            
    async def delete_document(self, document_id: str) -> bool:
        if document_id not in self.loaded_documents:
            return False
        
        success = await self._current_strategy.delete_document(document_id)
        if success:
            del self.loaded_documents[document_id]
        return success

    async def clear_all_documents(self) -> bool:
        success = await self._current_strategy.clear_all_documents()
        if success:
            self.loaded_documents.clear()
        return success