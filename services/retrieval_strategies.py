# services/retrieval_strategies.py
"""retrieval strategy implementations for RAG system"""

import logging
import uuid
from abc import ABC, abstractmethod
from typing import List, Any, Dict
from enum import Enum
import re
import asyncio

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions

from services.config import LOGGER_NAME, VECTOR_DB_PATH, EMBEDDING_MODEL_NAME

logger = logging.getLogger(LOGGER_NAME)

class RetrievalMethod(Enum):
    KEYWORD = "keyword"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"

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
    
    @abstractmethod
    def get_chunks_count(self) -> int:
        pass

class KeywordRetrieval(RetrievalStrategy):
    def __init__(self):
        self.documents: dict = {}
        logger.info("Keyword retrieval initialized")
    
    async def add_document(self, document_id: str, document_name: str, chunks: List[Any]) -> bool:
        try:
            self.documents[document_id] = {
                "name": document_name,
                "chunks": chunks
            }
            logger.info(f"Stored {len(chunks)} chunks for document '{document_name}'")
            return True
        except Exception as e:
            logger.error(f"Failed to add document to keyword retrieval: {e}")
            return False
    
    async def retrieve(self, question: str, top_k: int = 3) -> List[Dict]:
        if not self.documents:
            logger.error("No documents available for keyword retrieval")
            return []
        
        try:
            all_chunks = []
            for doc_id, doc_data in self.documents.items():
                for i, chunk in enumerate(doc_data['chunks']):
                    all_chunks.append({
                        "content": chunk.page_content,
                        "metadata": {
                            "document_id": doc_id,
                            "document_name": doc_data['name'],
                            "page_number": chunk.metadata.get("page", "N/A"),
                            "chunk_index": i
                        }
                    })

            question_words = set(re.findall(r'\w+', question.lower()))
            scored_chunks = []
            
            for chunk in all_chunks:
                content_words = set(re.findall(r'\w+', chunk['content'].lower()))
                
                intersection = question_words.intersection(content_words)
                if not intersection:
                    continue
                
                jaccard_score = len(intersection) / len(question_words.union(content_words)) if question_words.union(content_words) else 0
                word_count_score = len(intersection)
                
                final_score = (jaccard_score * 0.7) + (word_count_score * 0.3)
                
                scored_chunks.append((chunk, final_score))
            
            scored_chunks.sort(key=lambda x: x[1], reverse=True)
            relevant_chunks = [chunk for chunk, _ in scored_chunks[:top_k]]
            
            logger.debug(f"Keyword retrieval found {len(relevant_chunks)} relevant chunks")
            return relevant_chunks
            
        except Exception as e:
            logger.error(f"Keyword retrieval failed: {e}")
            raise
    
    async def delete_document(self, document_id: str) -> bool:
        try:
            if document_id in self.documents:
                del self.documents[document_id]
                logger.info(f"Deleted document '{document_id}' from keyword retrieval")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete document from keyword retrieval: {e}")
            return False

    async def clear_all_documents(self) -> bool:
        try:
            self.documents = {}
            logger.info("Cleared all chunks from keyword retrieval")
            return True
        except Exception as e:
            logger.error(f"Failed to clear keyword retrieval chunks: {e}")
            return False
    
    def get_chunks_count(self) -> int:
        return sum(len(doc['chunks']) for doc in self.documents.values())

class SemanticRetrieval(RetrievalStrategy):
    _embeddings_model_instance = None
    _client_instance = None

    def __init__(self):
        try:
            if SemanticRetrieval._embeddings_model_instance is None:
                SemanticRetrieval._embeddings_model_instance = SentenceTransformer(EMBEDDING_MODEL_NAME)
                logger.info("Loaded SentenceTransformer model instance")
            self.embeddings_model = SemanticRetrieval._embeddings_model_instance
            
            if SemanticRetrieval._client_instance is None:
                SemanticRetrieval._client_instance = chromadb.PersistentClient(path=VECTOR_DB_PATH)
                logger.info("Initialized persistent ChromaDB client instance")
            self.client = SemanticRetrieval._client_instance
            
            self.collection = self.client.get_or_create_collection(
                name="rag_chunks",
                embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=EMBEDDING_MODEL_NAME
                )
            )
            logger.info("Semantic retrieval initialized with persistent storage")

        except Exception as e:
            logger.error(f"Failed to initialize semantic retrieval: {e}")
            raise
    
    async def add_document(self, document_id: str, document_name: str, chunks: List[Any]) -> bool:
        try:
            texts = [chunk.page_content for chunk in chunks]
            metadatas = [
                {
                    "document_id": document_id, 
                    "document_name": document_name, 
                    "page": chunk.metadata.get("page", -1) + 1,
                }
                for chunk in chunks
            ]
            ids = [f"{document_id}_{i}" for i in range(len(chunks))]
            
            self.collection.add(
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Stored {len(texts)} vectors in ChromaDB for '{document_name}'")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add document to vector store: {e}")
            return False
    
    async def retrieve(self, question: str, top_k: int = 3) -> List[Dict]:
        if not self.collection.count():
            logger.warning("Vector store is empty")
            return []
        
        try:
            results = self.collection.query(
                query_texts=[question],
                n_results=top_k,
                include=['metadatas', 'documents']
            )
            
            retrieved_chunks = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    retrieved_chunks.append({
                        "content": results['documents'][0][i],
                        "metadata": results['metadatas'][0][i]
                    })
            
            logger.debug(f"Found {len(retrieved_chunks)} relevant chunks")
            return retrieved_chunks
                
        except Exception as e:
            logger.error(f"Semantic retrieval failed: {e}")
            raise
    
    async def delete_document(self, document_id: str) -> bool:
        try:
            self.collection.delete(where={"document_id": document_id})
            logger.info(f"Deleted document '{document_id}' and its chunks from ChromaDB")
            return True
        except Exception as e:
            logger.error(f"Failed to delete document '{document_id}' from ChromaDB: {e}")
            return False

    async def clear_all_documents(self) -> bool:
        try:
            self.client.delete_collection("rag_chunks")
            self.collection = self.client.create_collection(
                name="rag_chunks",
                embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=EMBEDDING_MODEL_NAME
                )
            )
            logger.info("Cleared all documents from vector store")
            return True
        except Exception as e:
            logger.error(f"Failed to clear documents: {e}")
            return False
    
    def get_chunks_count(self) -> int:
        try:
            return self.collection.count()
        except Exception as e:
            logger.error(f"Failed to get chunks count: {e}")
            return 0

class HybridRetrieval(RetrievalStrategy):
    def __init__(self, semantic_strategy: SemanticRetrieval):
        self.keyword_retrieval = KeywordRetrieval()
        self.semantic_retrieval = semantic_strategy
        logger.info("Hybrid retrieval initialized with keyword and semantic strategies")
    
    async def add_document(self, document_id: str, document_name: str, chunks: List[Any]) -> bool:
        try:
            keyword_success = await self.keyword_retrieval.add_document(document_id, document_name, chunks)
            semantic_success = await self.semantic_retrieval.add_document(document_id, document_name, chunks)
            
            if keyword_success and semantic_success:
                logger.info("Successfully added document to hybrid retrieval storage")
                return True
            else:
                logger.error("Failed to add document to one or both hybrid retrieval components")
                return False
                
        except Exception as e:
            logger.error(f"Failed to add document to hybrid retrieval: {e}")
            return False
    
    async def retrieve(self, question: str, top_k: int = 3) -> List[Dict]:
        try:
            keyword_results = await self.keyword_retrieval.retrieve(question, min(top_k * 2, 10))
            semantic_results = await self.semantic_retrieval.retrieve(question, min(top_k * 2, 10))
            
            chunk_scores = {}
            
            for i, chunk in enumerate(semantic_results):
                key = chunk['content']
                chunk_scores[key] = {
                    "score": chunk_scores.get(key, {}).get("score", 0) + 0.7 / (i + 1),
                    "metadata": chunk['metadata']
                }
            
            for i, chunk in enumerate(keyword_results):
                key = chunk['content']
                chunk_scores[key] = {
                    "score": chunk_scores.get(key, {}).get("score", 0) + 0.3 / (i + 1),
                    "metadata": chunk['metadata']
                }
            
            sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1]['score'], reverse=True)
            final_results = [
                {"content": chunk, "metadata": data['metadata']} 
                for chunk, data in sorted_chunks[:top_k]
            ]
            
            logger.debug(f"Hybrid retrieval combined {len(semantic_results)} semantic + {len(keyword_results)} keyword results into {len(final_results)} final results")
            return final_results
            
        except Exception as e:
            logger.error(f"Hybrid retrieval failed: {e}. Attempting fallback to semantic retrieval.")
            try:
                return await self.semantic_retrieval.retrieve(question, top_k)
            except Exception as e_fallback:
                logger.error(f"Semantic retrieval fallback also failed: {e_fallback}")
                return []
    
    async def delete_document(self, document_id: str) -> bool:
        try:
            keyword_cleared = await self.keyword_retrieval.delete_document(document_id)
            semantic_cleared = await self.semantic_retrieval.delete_document(document_id)
            success = keyword_cleared and semantic_cleared
            if success:
                logger.info(f"Deleted document '{document_id}' from hybrid retrieval")
            else:
                logger.error(f"Failed to delete document '{document_id}' from one or both hybrid components")
            return success
        except Exception as e:
            logger.error(f"Failed to delete hybrid retrieval documents: {e}")
            return False
            
    async def clear_all_documents(self) -> bool:
        try:
            keyword_cleared = await self.keyword_retrieval.clear_all_documents()
            semantic_cleared = await self.semantic_retrieval.clear_all_documents()
            success = keyword_cleared and semantic_cleared
            if success:
                logger.info("Cleared all documents from hybrid retrieval")
            else:
                logger.error("Failed to clear documents from one or both hybrid components")
            return success
        except Exception as e:
            logger.error(f"Failed to clear hybrid retrieval documents: {e}")
            return False
    
    def get_chunks_count(self) -> int:
        return self.semantic_retrieval.get_chunks_count()