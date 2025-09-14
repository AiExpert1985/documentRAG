# services/rag_service_refactored.py
"""Refactored RAG service with proper abstraction"""
import logging
from typing import List, Dict, Any
from pathlib import Path

from core.interfaces import (
    IRAGService, IVectorStore, 
    IEmbeddingService, IDocumentRepository, IMessageRepository,
    SearchResult
)
from services.document_processor_factory import DocumentProcessorFactory

logger = logging.getLogger(__name__)

class RAGService(IRAGService):
    """Main RAG service orchestrating all operations"""
    
    def __init__(
        self,
        vector_store: IVectorStore,
        doc_processor_factory: DocumentProcessorFactory, # <-- MODIFIED
        embedding_service: IEmbeddingService,
        document_repo: IDocumentRepository,
        message_repo: IMessageRepository
    ):
        self.vector_store = vector_store
        self.doc_processor_factory = doc_processor_factory # <-- MODIFIED
        self.embedding_service = embedding_service
        self.document_repo = document_repo
        self.message_repo = message_repo
    
    async def process_document(
        self, 
        file_path: str, 
        filename: str, 
        file_hash: str
    ) -> Dict[str, Any]:
        """Process and store a document"""
        try:
            # Check if document already exists
            existing = await self.document_repo.get_by_hash(file_hash)
            if existing:
                return {
                    "status": "error",
                    "error": "Document already exists"
                }
            
            # --- MODIFIED SECTION START ---
            # Get processor dynamically using the factory
            file_type = Path(filename).suffix[1:].lower()
            try:
                document_processor = self.doc_processor_factory.get_processor(file_type)
            except ValueError as e:
                return {"status": "error", "error": str(e)}

            # Validate document
            is_valid = await document_processor.validate(file_path, file_type)
            if not is_valid:
                return {
                    "status": "error",
                    "error": f"Invalid or corrupted {file_type} file"
                }
            # --- MODIFIED SECTION END ---
            
            # Create document record
            document = await self.document_repo.create(filename, file_hash)
            
            try:
                # Process document into chunks
                chunks = await document_processor.process(file_path, file_type)
                
                if not chunks:
                    await self.document_repo.delete(document.id)
                    return {"status": "error", "error": "No content could be extracted from the document."}

                # Update chunks with document ID
                for chunk in chunks:
                    chunk.document_id = document.id
                    chunk.metadata["document_id"] = document.id
                    chunk.metadata["document_name"] = filename
                
                # Generate embeddings
                texts = [chunk.content for chunk in chunks]
                embeddings = await self.embedding_service.generate_embeddings(texts)
                
                # Add embeddings to chunks
                for chunk, embedding in zip(chunks, embeddings):
                    chunk.embedding = embedding
                
                # Store in vector database
                success = await self.vector_store.add_chunks(chunks)
                if not success:
                    raise RuntimeError("Failed to store chunks in vector database")
                
                return {
                    "status": "success",
                    "filename": filename,
                    "document_id": document.id,
                    "chunks": len(chunks),
                    "pages": len(set(c.metadata.get("page", 1) for c in chunks))
                }
                
            except Exception as e:
                # Rollback document creation on failure
                await self.document_repo.delete(document.id)
                logger.error(f"Processing failed for doc {document.id}, rolling back. Error: {e}", exc_info=True)
                raise RuntimeError(f"Failed to process document content: {e}")
                
        except Exception as e:
            logger.error(f"Document processing failed for file '{filename}': {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Search across all documents"""
        try:
            # Generate query embedding
            query_embedding = await self.embedding_service.generate_query_embedding(query)
            
            # Search in vector store
            results = await self.vector_store.search(query_embedding, top_k)
            
            # Save search history
            await self.message_repo.save_search(query, len(results))
            
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    async def delete_document(self, document_id: str) -> bool:
        """Delete document and its chunks"""
        try:
            # Check document exists
            document = await self.document_repo.get_by_id(document_id)
            if not document:
                return False
            
            # Delete from vector store
            vector_deleted = await self.vector_store.delete_by_document(document_id)
            
            # Delete from database
            db_deleted = await self.document_repo.delete(document_id)
            
            return vector_deleted and db_deleted
            
        except Exception as e:
            logger.error(f"Document deletion failed: {e}")
            return False
    
    async def clear_all(self) -> bool:
        """Clear all documents and vectors"""
        try:
            vector_cleared = await self.vector_store.clear()
            db_cleared = await self.document_repo.delete_all()
            return vector_cleared and db_cleared
        except Exception as e:
            logger.error(f"Clear all failed: {e}")
            return False
    
    async def list_documents(self) -> List[Dict[str, str]]:
        """List all documents"""
        documents = await self.document_repo.list_all()
        return [
            {"id": doc.id, "filename": doc.filename}
            for doc in documents
        ]
    
    async def get_status(self) -> Dict[str, Any]:
        """Get system status"""
        documents = await self.document_repo.list_all()
        chunk_count = await self.vector_store.count()
        
        return {
            "document_loaded": ", ".join([d.filename for d in documents]) if documents else None,
            "chunks_available": chunk_count,
            "ready_for_queries": len(documents) > 0
        }