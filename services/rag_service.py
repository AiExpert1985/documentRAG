# services/rag_service.py
import logging
from typing import List, Dict, Any

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sqlalchemy import select, delete, func
from sqlalchemy.ext.asyncio import AsyncSession

from services.retrieval_strategies import RetrievalStrategy, SemanticRetrieval
from services.config import settings
from database.chat_db import Document

logger = logging.getLogger(settings.LOGGER_NAME)

class RAGService:
    """
    Orchestrates the RAG (Retrieval-Augmented Generation) pipeline.

    This service is stateless. It does not hold any document information in memory.
    All state is persisted in the database, which is accessed via the `db` session
    passed into its methods. It relies on a `RetrievalStrategy` for the actual
    vector search implementation.
    """

    def __init__(self, strategy: RetrievalStrategy):
        """
        Initializes the RAGService with a specific retrieval strategy.

        Args:
            strategy: An instance of a class that implements RetrievalStrategy.
        """
        self._current_strategy = strategy
        logger.info(f"RAG service initialized with strategy: {type(strategy).__name__}")

    async def has_documents(self, db: AsyncSession) -> bool:
        """Checks if any documents are present in the database."""
        result = await db.execute(select(func.count(Document.id)))
        return result.scalar_one() > 0

    async def get_chunks_count(self) -> int:
        """
        Gets the total number of chunks from the retrieval strategy's vector store.
        """
        # Since we only use SemanticRetrieval, we can access its collection directly.
        if hasattr(self._current_strategy, 'collection'):
            try:
                return self._current_strategy.collection.count()
            except Exception as e:
                logger.error(f"Could not get chunk count from vector store: {e}")
        return 0

    async def process_pdf_file(self, db: AsyncSession, file_path: str, original_filename: str, file_hash: str) -> Dict[str, Any]:
        """
        Processes a PDF file, creates a database entry, and ingests it into the vector store.
        This operation is transactional: if vector ingestion fails, the DB entry is rolled back.
        """
        document_id = None
        try:
            # 1. Create a DB entry for the document.
            new_doc = Document(filename=original_filename, file_hash=file_hash)
            db.add(new_doc)
            await db.commit()
            await db.refresh(new_doc)
            document_id = new_doc.id

            # 2. Process the PDF and split it into chunks.
            logger.info(f"Loading and splitting PDF: {original_filename}")
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(documents)
            
            if not chunks:
                 raise ValueError("PDF processing resulted in zero chunks.")

            # 3. Add the document chunks to the vector store.
            logger.info(f"Adding {len(chunks)} chunks to vector store for document ID: {document_id}")
            success = await self._current_strategy.add_document(document_id, original_filename, chunks)
            
            if not success:
                # This will trigger the exception block to roll back the DB operation.
                raise RuntimeError("Failed to store document chunks in the vector database.")
            
            logger.info(f"Successfully processed and ingested document: {original_filename}")
            return {
                "status": "success", "filename": original_filename,
                "pages": len(documents), "chunks": len(chunks)
            }
        except Exception as e:
            logger.error(f"Error processing PDF '{original_filename}': {e}", exc_info=True)
            # If any step fails, roll back the database commit.
            if document_id:
                logger.warning(f"Rolling back database entry for document ID: {document_id}")
                await db.execute(delete(Document).where(Document.id == document_id))
                await db.commit()
            return {"error": str(e), "status": "error"}

    async def retrieve_chunks(self, db: AsyncSession, question: str, top_k: int = 3) -> List[Dict]:
        """Retrieves relevant chunks for a given question."""
        if not await self.has_documents(db):
            logger.warning("Attempted to retrieve chunks, but no documents are loaded.")
            return []
        
        logger.info(f"Retrieving top {top_k} chunks for question.")
        return await self._current_strategy.retrieve(question, top_k)

    async def list_documents(self, db: AsyncSession) -> List[Dict[str, str]]:
        """Lists all documents currently stored in the database."""
        result = await db.execute(select(Document).order_by(Document.timestamp.desc()))
        docs = result.scalars().all()
        return [{"id": doc.id, "filename": doc.filename} for doc in docs]

    async def delete_document(self, db: AsyncSession, document_id: str) -> bool:
        """
        Deletes a document from both the vector store and the SQL database.
        """
        doc_to_delete = await db.get(Document, document_id)
        if not doc_to_delete:
            logger.warning(f"Attempted to delete non-existent document with ID: {document_id}")
            return False
        
        logger.info(f"Deleting document from vector store: ID {document_id}")
        success = await self._current_strategy.delete_document(document_id)
        
        if success:
            logger.info(f"Deleting document from database: ID {document_id}")
            await db.delete(doc_to_delete)
            await db.commit()
            return True
        
        logger.error(f"Failed to delete document from vector store: ID {document_id}. Database entry will not be removed.")
        return False

    async def clear_all_documents(self, db: AsyncSession) -> bool:
        """
        Clears all documents from the vector store and the SQL database.
        """
        logger.info("Clearing all documents from vector store.")
        success = await self._current_strategy.clear_all_documents()
        
        if success:
            logger.info("Clearing all document entries from the database.")
            await db.execute(delete(Document))
            await db.commit()
            return True

        logger.error("Failed to clear documents from vector store. Database entries will not be removed.")
        return False