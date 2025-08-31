# services/search_strategies.py
"""Search strategy implementations for RAG system"""

import logging
from abc import ABC, abstractmethod
from typing import List, Any
from enum import Enum

# Required dependencies
from sentence_transformers import SentenceTransformer
import chromadb

# Setup logging
logger = logging.getLogger(__name__)

class SearchMethod(Enum):
    KEYWORD = "keyword"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"

class SearchStrategy(ABC):
    """Abstract base class for different search strategies"""
    
    @abstractmethod
    def setup_document_store(self, chunks: List[Any]) -> bool:
        """Setup storage for document chunks"""
        pass
    
    @abstractmethod
    def search(self, question: str, top_k: int = 3) -> List[str]:
        """Search for relevant chunks and return content"""
        pass
    
    @abstractmethod
    def clear_documents(self) -> bool:
        """Clear all stored documents"""
        pass
    
    @abstractmethod
    def get_chunks_count(self) -> int:
        """Get total number of stored chunks"""
        pass

class KeywordSearch(SearchStrategy):
    """Simple keyword-based search strategy"""
    
    def __init__(self):
        self.chunks: List[Any] = []
        logger.info("Keyword search initialized")
    
    def setup_document_store(self, chunks: List[Any]) -> bool:
        """Store chunks in memory for keyword search"""
        try:
            self.chunks = chunks
            logger.info(f"Stored {len(chunks)} chunks for keyword search")
            return True
        except Exception as e:
            logger.error(f"Failed to setup keyword search: {e}")
            return False
    
    def search(self, question: str, top_k: int = 3) -> List[str]:
        """Search for keyword matches in chunks"""
        if not self.chunks:
            logger.error("No chunks available for keyword search")
            raise RuntimeError("No chunks loaded for keyword search")
        
        try:
            relevant_chunks: List[str] = []
            question_lower: str = question.lower()
            
            # Score chunks by keyword matches
            scored_chunks: List[tuple] = []
            
            for chunk in self.chunks:
                content_lower = chunk.page_content.lower()
                score = sum(1 for word in question_lower.split() 
                           if word in content_lower)
                
                if score > 0:
                    scored_chunks.append((chunk.page_content, score))
            
            # Sort by score and take top_k
            scored_chunks.sort(key=lambda x: x[1], reverse=True)
            relevant_chunks = [content for content, _ in scored_chunks[:top_k]]
            
            # Fallback: return first chunk if no matches
            if not relevant_chunks and self.chunks:
                relevant_chunks = [self.chunks[0].page_content]
            
            logger.debug(f"Keyword search found {len(relevant_chunks)} relevant chunks")
            return relevant_chunks
            
        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
            raise
    
    def clear_documents(self) -> bool:
        """Clear all stored chunks"""
        try:
            self.chunks = []
            logger.info("Cleared all chunks from keyword search")
            return True
        except Exception as e:
            logger.error(f"Failed to clear keyword search chunks: {e}")
            return False
    
    def get_chunks_count(self) -> int:
        """Get number of stored chunks"""
        return len(self.chunks)
    """Semantic similarity-based search using embeddings"""
    
    def __init__(self):
        try:
            self.embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.client = chromadb.PersistentClient(path="./vector_db")
            self.collection = None
            logger.info("Semantic search initialized with persistent storage")
        except Exception as e:
            logger.error(f"Failed to initialize semantic search: {e}")
            raise
    
    def setup_document_store(self, chunks: List[Any]) -> bool:
        """Setup vector database with chunk embeddings"""
        try:
            collection_name = "rag_chunks"
            
            # Clear existing collection for single document mode
            try:
                self.client.delete_collection(collection_name)
                logger.info(f"Cleared existing collection: {collection_name}")
            except Exception:
                pass  # Collection might not exist
            
            # Create fresh collection
            self.collection = self.client.create_collection(collection_name)
            logger.info(f"Created new collection: {collection_name}")
            
            # Generate embeddings for all chunks
            texts = [chunk.page_content for chunk in chunks]
            logger.info(f"Generating embeddings for {len(texts)} chunks...")
            
            embeddings = self.embeddings_model.encode(texts)
            logger.info(f"Generated {len(embeddings)} embeddings")
            
            # Store in vector database
            for i, (text, embedding) in enumerate(zip(texts, embeddings)):
                self.collection.add(
                    embeddings=[embedding.tolist()],
                    documents=[text],
                    ids=[f"chunk_{i}"]
                )
            
            logger.info(f"Stored {len(embeddings)} vectors in ChromaDB")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup vector store: {e}")
            return False
    
    def search(self, question: str, top_k: int = 3) -> List[str]:
        """Search for semantically similar chunks"""
        if not self.collection:
            logger.error("Vector store not initialized")
            raise RuntimeError("Vector store not initialized. Call setup_document_store first.")
        
        try:
            # Generate query embedding
            query_embedding = self.embeddings_model.encode([question])
            
            # Search similar chunks
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=top_k
            )
            
            # Return the most relevant chunks
            if results['documents'] and results['documents'][0]:
                logger.debug(f"Found {len(results['documents'][0])} relevant chunks")
                return results['documents'][0]
            else:
                logger.warning("No relevant chunks found")
                return []
                
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            raise
    
    def clear_documents(self) -> bool:
        """Clear all stored documents"""
        try:
            if self.collection:
                self.client.delete_collection("rag_chunks")
                self.collection = None
                logger.info("Cleared all documents from vector store")
                return True
            return True
        except Exception as e:
            logger.error(f"Failed to clear documents: {e}")
            return False
    
    def get_chunks_count(self) -> int:
        """Get number of chunks in vector store"""
        try:
            if self.collection:
                return self.collection.count()
            return 0
        except Exception as e:
            logger.error(f"Failed to get chunks count: {e}")
            return 0

class HybridSearch(SearchStrategy):
    """Combines keyword and semantic search for better results"""
    
    def __init__(self):
        self.keyword_search = KeywordSearch()
        self.semantic_search = SemanticSearch()
        logger.info("Hybrid search initialized with keyword and semantic strategies")
    
    def setup_document_store(self, chunks: List[Any]) -> bool:
        """Setup both keyword and semantic stores"""
        try:
            keyword_success = self.keyword_search.setup_document_store(chunks)
            semantic_success = self.semantic_search.setup_document_store(chunks)
            
            if keyword_success and semantic_success:
                logger.info("Successfully setup hybrid search storage")
                return True
            else:
                logger.error("Failed to setup one or both hybrid search components")
                return False
                
        except Exception as e:
            logger.error(f"Failed to setup hybrid search: {e}")
            return False
    
    def search(self, question: str, top_k: int = 3) -> List[str]:
        """Combine keyword and semantic search results"""
        try:
            # Get results from both strategies
            keyword_results = self.keyword_search.search(question, top_k)
            semantic_results = self.semantic_search.search(question, top_k)
            
            # Combine and deduplicate results
            combined_results = []
            seen_chunks = set()
            
            # Add semantic results first (usually higher quality)
            for chunk in semantic_results:
                chunk_hash = hash(chunk[:100])  # Use first 100 chars as identifier
                if chunk_hash not in seen_chunks:
                    combined_results.append(chunk)
                    seen_chunks.add(chunk_hash)
            
            # Add keyword results that aren't already included
            for chunk in keyword_results:
                chunk_hash = hash(chunk[:100])
                if chunk_hash not in seen_chunks and len(combined_results) < top_k:
                    combined_results.append(chunk)
                    seen_chunks.add(chunk_hash)
            
            logger.debug(f"Hybrid search combined {len(semantic_results)} semantic + {len(keyword_results)} keyword results into {len(combined_results)} final results")
            return combined_results[:top_k]
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            # Fallback to semantic search only
            return self.semantic_search.search(question, top_k)
    
    def clear_documents(self) -> bool:
        """Clear documents from both strategies"""
        try:
            keyword_cleared = self.keyword_search.clear_documents()
            semantic_cleared = self.semantic_search.clear_documents()
            
            success = keyword_cleared and semantic_cleared
            if success:
                logger.info("Cleared documents from hybrid search")
            else:
                logger.error("Failed to clear documents from one or both hybrid components")
            
            return success
        except Exception as e:
            logger.error(f"Failed to clear hybrid search documents: {e}")
            return False
    
    def get_chunks_count(self) -> int:
        """Get chunks count from semantic search (primary store)"""
        return self.semantic_search.get_chunks_count()

# Future search strategies can be added here following the same pattern