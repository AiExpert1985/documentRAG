# infrastructure/vector_stores.py
"""Concrete implementations of vector stores"""
import asyncio
import logging
from typing import List, Optional
from chromadb import Client
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from core.interfaces import IVectorStore, DocumentChunk, SearchResult

logger = logging.getLogger(__name__)

class ChromaDBVectorStore(IVectorStore):
    """ChromaDB implementation of vector store"""
    
    def __init__(self, client: Client, collection_name: str = "rag_chunks"):
        self._client = client
        self._collection_name = collection_name
        self._collection = None
        
    async def _ensure_collection(self):
        """Lazy initialization of collection"""
        if not self._collection:
            self._collection = await asyncio.to_thread(
                self._client.get_or_create_collection,
                name=self._collection_name
            )
    
    async def add_chunks(self, chunks: List[DocumentChunk]) -> bool:
        """Add chunks to ChromaDB"""
        try:
            await self._ensure_collection()
            
            if not chunks:
                return True
                
            texts = [chunk.content for chunk in chunks]
            metadatas = [chunk.metadata for chunk in chunks]
            ids = [chunk.id for chunk in chunks]
            embeddings = [chunk.embedding for chunk in chunks if chunk.embedding]
            
            await asyncio.to_thread(
                self._collection.add,
                documents=texts,
                metadatas=metadatas,
                ids=ids,
                embeddings=embeddings if embeddings else None
            )
            return True
            
        except Exception as e:
            logger.error(f"Failed to add chunks to ChromaDB: {e}")
            return False
    
    async def search(self, query_embedding: List[float], top_k: int = 5) -> List[SearchResult]:
        """Search for similar chunks"""
        try:
            await self._ensure_collection()
            
            results = await asyncio.to_thread(
                self._collection.query,
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=['metadatas', 'documents', 'distances']
            )
            
            search_results = []
            if results['ids'] and results['ids'][0]:
                for i in range(len(results['ids'][0])):
                    chunk = DocumentChunk(
                        id=results['ids'][0][i],
                        content=results['documents'][0][i],
                        document_id=results['metadatas'][0][i].get('document_id'),
                        metadata=results['metadatas'][0][i]
                    )
                    search_results.append(SearchResult(
                        chunk=chunk,
                        score=1 - results['distances'][0][i]  # Convert distance to similarity
                    ))
            
            return search_results
            
        except Exception as e:
            logger.error(f"Search failed in ChromaDB: {e}")
            return []
    
    async def delete_by_document(self, document_id: str) -> bool:
        """Delete all chunks for a document"""
        try:
            await self._ensure_collection()
            await asyncio.to_thread(
                self._collection.delete,
                where={"document_id": document_id}
            )
            return True
        except Exception as e:
            logger.error(f"Failed to delete document chunks: {e}")
            return False
    
    async def clear(self) -> bool:
        """Clear all vectors"""
        try:
            await asyncio.to_thread(
                self._client.delete_collection,
                name=self._collection_name
            )
            self._collection = None
            return True
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            return False
    
    async def count(self) -> int:
        """Get chunk count"""
        try:
            await self._ensure_collection()
            return await asyncio.to_thread(self._collection.count)
        except Exception as e:
            logger.error(f"Failed to get count: {e}")
            return 0