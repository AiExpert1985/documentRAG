# services/retrieval_strategies.py
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import List, Any, Dict

from chromadb import Client, Collection
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from config import settings

logger = logging.getLogger(settings.LOGGER_NAME)

class RetrievalStrategy(ABC):
    @abstractmethod
    async def add_document(self, document_id: str, document_name: str, chunks: List[Any]) -> bool:
        pass
    
    @abstractmethod
    async def retrieve(self, question: str, top_k: int = 3) -> List[Dict]:
        pass
        
    @abstractmethod
    async def delete_document(self, document_id: str) -> bool:
        pass
    
    @abstractmethod
    async def clear_all_documents(self) -> bool:
        pass

class SemanticRetrieval(RetrievalStrategy):
    def __init__(self, client: Client, embedding_function: SentenceTransformerEmbeddingFunction):
        self._client = client
        self._embedding_function = embedding_function
        self.collection_name = "rag_chunks"
        self.collection: Collection = self._client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self._embedding_function
        )
    
    async def add_document(self, document_id: str, document_name: str, chunks: List[Any]) -> bool:
        try:
            texts = [chunk.page_content for chunk in chunks]
            metadatas = [
                {"document_id": document_id, "document_name": document_name, "page": chunk.metadata.get("page", -1) + 1}
                for chunk in chunks
            ]
            ids = [f"{document_id}_{i}" for i in range(len(chunks))]
            
            # Run the blocking 'add' method in a separate thread
            await asyncio.to_thread(
                self.collection.add, 
                documents=texts, 
                metadatas=metadatas, 
                ids=ids
            )
            return True
        except Exception as e:
            logger.error(f"Failed to add document to ChromaDB: {e}", exc_info=True)
            return False
    
    async def retrieve(self, question: str, top_k: int = 3) -> List[Dict]:
        try:
            # Run the blocking 'query' method in a separate thread
            results = await asyncio.to_thread(
                self.collection.query,
                query_texts=[question], 
                n_results=top_k, 
                include=['metadatas', 'documents']
            )
            
            retrieved_chunks = []
            if results['ids'] and results['ids'][0]:
                for i in range(len(results['ids'][0])):
                    retrieved_chunks.append({
                        "id": results['ids'][0][i],
                        "content": results['documents'][0][i],
                        "metadata": results['metadatas'][0][i]
                    })
            return retrieved_chunks
        except Exception as e:
            logger.error(f"Semantic retrieval failed: {e}", exc_info=True)
            return []
    
    async def delete_document(self, document_id: str) -> bool:
        try:
            # Run the blocking 'delete' method in a separate thread
            await asyncio.to_thread(
                self.collection.delete, 
                where={"document_id": document_id}
            )
            return True
        except Exception as e:
            logger.error(f"Delete document from ChromaDB failed: {e}", exc_info=True)
            return False

    async def clear_all_documents(self) -> bool:
        try:
            # This is a safer way to clear and reset the collection state
            await asyncio.to_thread(self._client.delete_collection, name=self.collection_name)
            
            self.collection = await asyncio.to_thread(
                self._client.get_or_create_collection,
                name=self.collection_name,
                embedding_function=self._embedding_function
            )
            logger.info(f"ChromaDB collection '{self.collection_name}' cleared and re-created successfully.")
            return True
        except Exception as e:
            logger.error(f"Clear all from ChromaDB failed: {e}", exc_info=True)
            return False