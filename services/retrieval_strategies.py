# services/retrieval_strategies.py
import logging
from abc import ABC, abstractmethod
from typing import List, Any, Dict
from enum import Enum
import re
from collections import defaultdict

from chromadb import Client, Collection
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from services.config import settings

logger = logging.getLogger(settings.LOGGER_NAME)

class RetrievalMethod(Enum):
    KEYWORD = "keyword"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"

def reciprocal_rank_fusion(results: List[List[Dict]], k: int = 60) -> List[Dict]:
    """
    Performs Reciprocal Rank Fusion on multiple lists of search results.
    Each result dictionary must have a unique 'id' field.
    """
    ranked_lists = [
        {item['id']: 1 / (rank + k) for rank, item in enumerate(sublist)}
        for sublist in results
    ]
    
    fused_scores = defaultdict(float)
    for r_list in ranked_lists:
        for doc_id, score in r_list.items():
            fused_scores[doc_id] += score
            
    # Create a master dictionary of all unique documents
    all_docs = {}
    for sublist in results:
        for doc in sublist:
            all_docs[doc['id']] = doc

    reranked_results = [
        all_docs[doc_id]
        for doc_id, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    
    return reranked_results

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

class KeywordRetrieval(RetrievalStrategy):
    """
    NOTE: This is a placeholder for a robust keyword search.
    The previous in-memory implementation was not scalable.
    For production, this should be replaced with a real sparse index like BM25.
    """
    def __init__(self):
        logger.warning("KeywordRetrieval is using a non-scalable placeholder implementation.")

    async def add_document(self, document_id: str, document_name: str, chunks: List[Any]) -> bool:
        return True # Placeholder
    
    async def retrieve(self, question: str, top_k: int = 3) -> List[Dict]:
        return [] # Placeholder
    
    async def delete_document(self, document_id: str) -> bool:
        return True # Placeholder

    async def clear_all_documents(self) -> bool:
        return True # Placeholder

class SemanticRetrieval(RetrievalStrategy):
    def __init__(self, client: Client, embedding_function: SentenceTransformerEmbeddingFunction):
        self.collection: Collection = client.get_or_create_collection(
            name="rag_chunks",
            embedding_function=embedding_function
        )
    
    async def add_document(self, document_id: str, document_name: str, chunks: List[Any]) -> bool:
        try:
            texts = [chunk.page_content for chunk in chunks]
            metadatas = [
                {"document_id": document_id, "document_name": document_name, "page": chunk.metadata.get("page", -1) + 1}
                for chunk in chunks
            ]
            ids = [f"{document_id}_{i}" for i in range(len(chunks))]
            
            self.collection.add(documents=texts, metadatas=metadatas, ids=ids)
            return True
        except Exception as e:
            logger.error(f"Failed to add document to ChromaDB: {e}")
            return False
    
    async def retrieve(self, question: str, top_k: int = 3) -> List[Dict]:
        try:
            results = self.collection.query(query_texts=[question], n_results=top_k, include=['metadatas', 'documents'])
            
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
            logger.error(f"Semantic retrieval failed: {e}")
            return []
    
    async def delete_document(self, document_id: str) -> bool:
        try:
            self.collection.delete(where={"document_id": document_id})
            return True
        except Exception as e:
            logger.error(f"Delete document from ChromaDB failed: {e}")
            return False

    async def clear_all_documents(self) -> bool:
        try:
            # Re-creating the collection is a reliable way to clear it
            self.collection._client.delete_collection(self.collection.name)
            self.collection = self.collection._client.create_collection(
                name=self.collection.name,
                embedding_function=self.collection._embedding_function
            )
            return True
        except Exception as e:
            logger.error(f"Clear all from ChromaDB failed: {e}")
            return False

class HybridRetrieval(RetrievalStrategy):
    def __init__(self, semantic_strategy: SemanticRetrieval, keyword_strategy: KeywordRetrieval):
        self.semantic = semantic_strategy
        self.keyword = keyword_strategy
    
    async def add_document(self, document_id: str, document_name: str, chunks: List[Any]) -> bool:
        sem_success = await self.semantic.add_document(document_id, document_name, chunks)
        kw_success = await self.keyword.add_document(document_id, document_name, chunks)
        return sem_success and kw_success
    
    async def retrieve(self, question: str, top_k: int = 3) -> List[Dict]:
        # Fetch more results from each retriever to provide a good pool for fusion
        kw_results = await self.keyword.retrieve(question, top_k * 2)
        sem_results = await self.semantic.retrieve(question, top_k * 2)
        
        fused_results = reciprocal_rank_fusion([sem_results, kw_results])
        
        return fused_results[:top_k]
    
    async def delete_document(self, document_id: str) -> bool:
        sem_success = await self.semantic.delete_document(document_id)
        kw_success = await self.keyword.delete_document(document_id)
        return sem_success and kw_success
            
    async def clear_all_documents(self) -> bool:
        sem_success = await self.semantic.clear_all_documents()
        kw_success = await self.keyword.clear_all_documents()
        return sem_success and kw_success