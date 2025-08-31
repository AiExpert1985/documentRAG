# services/retrieval_strategies.py
import logging
from abc import ABC, abstractmethod
from typing import List, Any, Dict, Optional
from enum import Enum
import re

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
    
    async def add_document(self, document_id: str, document_name: str, chunks: List[Any]) -> bool:
        try:
            self.documents[document_id] = {"name": document_name, "chunks": chunks}
            return True
        except Exception as e:
            logger.error(f"Failed to add document: {e}")
            return False
    
    async def retrieve(self, question: str, top_k: int = 3) -> List[Dict]:
        if not self.documents:
            return []
        
        all_chunks = []
        for doc_id, doc_data in self.documents.items():
            for i, chunk in enumerate(doc_data['chunks']):
                all_chunks.append({
                    "content": chunk.page_content,
                    "metadata": {
                        "document_id": doc_id,
                        "document_name": doc_data['name'],
                        "page": chunk.metadata.get("page", "N/A"),
                        "chunk_index": i
                    }
                })

        question_words = set(re.findall(r'\w+', question.lower()))
        scored_chunks = []
        
        for chunk in all_chunks:
            content_words = set(re.findall(r'\w+', chunk['content'].lower()))
            intersection = question_words.intersection(content_words)
            
            if intersection:
                score = len(intersection) / len(question_words)
                scored_chunks.append((chunk, score))
        
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return [chunk for chunk, _ in scored_chunks[:top_k]]
    
    async def delete_document(self, document_id: str) -> bool:
        if document_id in self.documents:
            del self.documents[document_id]
            return True
        return False

    async def clear_all_documents(self) -> bool:
        self.documents = {}
        return True
    
    def get_chunks_count(self) -> int:
        return sum(len(doc['chunks']) for doc in self.documents.values())

class SemanticRetrieval(RetrievalStrategy):
    _model = None
    _client = None

    def __init__(self):
        if SemanticRetrieval._model is None:
            SemanticRetrieval._model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        
        if SemanticRetrieval._client is None:
            SemanticRetrieval._client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
        
        self.collection = SemanticRetrieval._client.get_or_create_collection(
            name="rag_chunks",
            embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=EMBEDDING_MODEL_NAME
            )
        )
    
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
            
            self.collection.add(documents=texts, metadatas=metadatas, ids=ids)
            return True
        except Exception as e:
            logger.error(f"Failed to add document: {e}")
            return False
    
    async def retrieve(self, question: str, top_k: int = 3) -> List[Dict]:
        if not self.collection.count():
            return []
        
        try:
            results = self.collection.query(
                query_texts=[question], n_results=top_k,
                include=['metadatas', 'documents']
            )
            
            chunks = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    chunks.append({
                        "content": results['documents'][0][i],
                        "metadata": results['metadatas'][0][i]
                    })
            return chunks
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return []
    
    async def delete_document(self, document_id: str) -> bool:
        try:
            self.collection.delete(where={"document_id": document_id})
            return True
        except Exception as e:
            logger.error(f"Delete failed: {e}")
            return False

    async def clear_all_documents(self) -> bool:
        try:
            SemanticRetrieval._client.delete_collection("rag_chunks")
            self.collection = SemanticRetrieval._client.create_collection(
                name="rag_chunks",
                embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=EMBEDDING_MODEL_NAME
                )
            )
            return True
        except Exception as e:
            logger.error(f"Clear failed: {e}")
            return False
    
    def get_chunks_count(self) -> int:
        try:
            return self.collection.count()
        except:
            return 0

class HybridRetrieval(RetrievalStrategy):
    def __init__(self):
        self.keyword = KeywordRetrieval()
        self.semantic = SemanticRetrieval()
    
    async def add_document(self, document_id: str, document_name: str, chunks: List[Any]) -> bool:
        kw_success = await self.keyword.add_document(document_id, document_name, chunks)
        sem_success = await self.semantic.add_document(document_id, document_name, chunks)
        return kw_success and sem_success
    
    async def retrieve(self, question: str, top_k: int = 3) -> List[Dict]:
        kw_results = await self.keyword.retrieve(question, top_k * 2)
        sem_results = await self.semantic.retrieve(question, top_k * 2)
        
        # Simple combination - semantic results get higher weight
        combined = {}
        for i, chunk in enumerate(sem_results):
            combined[chunk['content'][:100]] = {"chunk": chunk, "score": 0.7 / (i + 1)}
        
        for i, chunk in enumerate(kw_results):
            key = chunk['content'][:100]
            if key in combined:
                combined[key]["score"] += 0.3 / (i + 1)
            else:
                combined[key] = {"chunk": chunk, "score": 0.3 / (i + 1)}
        
        sorted_results = sorted(combined.values(), key=lambda x: x['score'], reverse=True)
        return [item['chunk'] for item in sorted_results[:top_k]]
    
    async def delete_document(self, document_id: str) -> bool:
        kw_success = await self.keyword.delete_document(document_id)
        sem_success = await self.semantic.delete_document(document_id)
        return kw_success and sem_success
            
    async def clear_all_documents(self) -> bool:
        kw_success = await self.keyword.clear_all_documents()
        sem_success = await self.semantic.clear_all_documents()
        return kw_success and sem_success
    
    def get_chunks_count(self) -> int:
        return self.semantic.get_chunks_count()