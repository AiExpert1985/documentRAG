# services/rag_service.py
import asyncio
import logging
from typing import List, Dict, Any

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sqlalchemy import select, delete, func
from sqlalchemy.ext.asyncio import AsyncSession
from pypdf.errors import PdfReadError

from services.retrieval_strategies import RetrievalStrategy
from config import settings
from database.chat_db import Document

logger = logging.getLogger(settings.LOGGER_NAME)

class RAGService:
    def __init__(self, strategy: RetrievalStrategy):
        self._current_strategy = strategy
        logger.info(f"RAG service initialized with strategy: {type(strategy).__name__}")

    async def has_documents(self, db: AsyncSession) -> bool:
        result = await db.execute(select(func.count(Document.id)))
        return result.scalar_one() > 0

    async def get_chunks_count(self) -> int:
        if hasattr(self._current_strategy, 'collection'):
            try:
                # collection.count() is a blocking call
                return await asyncio.to_thread(self._current_strategy.collection.count)
            except Exception as e:
                logger.error(f"Could not get chunk count from vector store: {e}")
        return 0

    async def process_pdf_file(self, db: AsyncSession, file_path: str, original_filename: str, file_hash: str) -> Dict[str, Any]:
        document_id = None
        try:
            new_doc = Document(filename=original_filename, file_hash=file_hash)
            db.add(new_doc)
            await db.commit()
            await db.refresh(new_doc)
            document_id = new_doc.id

            logger.info(f"Loading and splitting PDF: {original_filename}")
            loader = PyPDFLoader(file_path)
            # Run the blocking 'load' method in a separate thread
            documents = await asyncio.to_thread(loader.load)
            
            if not documents:
                raise ValueError("No content could be extracted from the PDF. It may be empty or image-based.")

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            # Run the blocking 'split_documents' method in a separate thread
            chunks = await asyncio.to_thread(text_splitter.split_documents, documents)
            
            if not chunks:
                 raise ValueError("PDF content could not be split into chunks.")

            logger.info(f"Adding {len(chunks)} chunks to vector store for document ID: {document_id}")
            success = await self._current_strategy.add_document(document_id, original_filename, chunks)
            
            if not success:
                raise RuntimeError("Failed to store document chunks in the vector database.")
            
            return {"status": "success", "filename": original_filename, "pages": len(documents), "chunks": len(chunks)}

        except PdfReadError:
            logger.error(f"Failed to read PDF '{original_filename}'. It may be corrupted or encrypted.")
            if document_id:
                await db.execute(delete(Document).where(Document.id == document_id))
                await db.commit()
            return {"error": "Failed to read the PDF. The file may be corrupted or encrypted.", "status": "error"}
        
        except Exception as e:
            logger.error(f"Error processing PDF '{original_filename}': {e}", exc_info=True)
            if document_id:
                await db.execute(delete(Document).where(Document.id == document_id))
                await db.commit()
            return {"error": str(e), "status": "error"}

    async def retrieve_chunks(self, db: AsyncSession, question: str, top_k: int = 3) -> List[Dict]:
        if not await self.has_documents(db):
            return []
        return await self._current_strategy.retrieve(question, top_k)

    async def list_documents(self, db: AsyncSession) -> List[Dict[str, str]]:
        result = await db.execute(select(Document).order_by(Document.timestamp.desc()))
        docs = result.scalars().all()
        return [{"id": doc.id, "filename": doc.filename} for doc in docs]

    async def delete_document(self, db: AsyncSession, document_id: str) -> bool:
        doc_to_delete = await db.get(Document, document_id)
        if not doc_to_delete:
            return False
        
        success = await self._current_strategy.delete_document(document_id)
        if success:
            await db.delete(doc_to_delete)
            await db.commit()
        return success

    async def clear_all_documents(self, db: AsyncSession) -> bool:
        success = await self._current_strategy.clear_all_documents()
        if success:
            await db.execute(delete(Document))
            await db.commit()
        return success