# services/search_strategies.py
"""Search strategy implementations for RAG system"""

from abc import ABC, abstractmethod
from typing import List, Any, Tuple
from enum import Enum

# First install: pip install sentence-transformers chromadb
try:
    from sentence_transformers import SentenceTransformer
    import chromadb
    SEMANTIC_SEARCH_AVAILABLE = True
except ImportError:
    SEMANTIC_SEARCH_AVAILABLE = False

class SearchMethod(Enum):
    KEYWORD = "keyword"
    SEMANTIC = "semantic"

class SearchStrategy(ABC):
    """Abstract base class for different search strategies"""
    
    @abstractmethod
    def search(self, question: str, chunks: List[Any], top_k: int = 3) -> List[str]:
        """Search for relevant chunks and return content"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this search method is available"""
        pass

class KeywordSearch(SearchStrategy):
    """Simple keyword-based search strategy"""
    
    def search(self, question: str, chunks: List[Any], top_k: int = 3) -> List[str]:
        relevant_chunks: List[str] = []
        question_lower: str = question.lower()
        
        # Score chunks by keyword matches
        scored_chunks: List[Tuple[str, int]] = []
        
        for chunk in chunks:
            content_lower = chunk.page_content.lower()
            score = sum(1 for word in question_lower.split() 
                       if word in content_lower)
            
            if score > 0:
                scored_chunks.append((chunk.page_content, score))
        
        # Sort by score and take top_k
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        relevant_chunks = [content for content, _ in scored_chunks[:top_k]]
        
        # Fallback: return first chunk if no matches
        if not relevant_chunks and chunks:
            relevant_chunks = [chunks[0].page_content]
        
        return relevant_chunks
    
    def is_available(self) -> bool:
        return True

class SemanticSearch(SearchStrategy):
    """Semantic similarity-based search using embeddings"""
    
    def __init__(self):
        self.embeddings_model = None
        self.vector_db = None
        self.collection = None
        
        if SEMANTIC_SEARCH_AVAILABLE:
            try:
                self.embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
                self.client = chromadb.Client()
            except Exception as e:
                print(f"Failed to initialize semantic search: {e}")
    
    def setup_vector_store(self, chunks: List[Any]) -> bool:
        """Setup vector database with chunk embeddings"""
        if not self.is_available():
            return False
        
        try:
            # Create new collection (delete if exists)
            try:
                self.client.delete_collection("rag_chunks")
            except:
                pass
            
            self.collection = self.client.create_collection("rag_chunks")
            
            # Generate embeddings for all chunks
            texts = [chunk.page_content for chunk in chunks]
            embeddings = self.embeddings_model.encode(texts)
            
            # Store in vector database
            for i, (text, embedding) in enumerate(zip(texts, embeddings)):
                self.collection.add(
                    embeddings=[embedding.tolist()],
                    documents=[text],
                    ids=[f"chunk_{i}"]
                )
            
            return True
            
        except Exception as e:
            print(f"Failed to setup vector store: {e}")
            return False
    
    def search(self, question: str, chunks: List[Any], top_k: int = 3) -> List[str]:
        if not self.is_available() or not self.collection:
            # Fallback to keyword search
            keyword_search = KeywordSearch()
            return keyword_search.search(question, chunks, top_k)
        
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
                return results['documents'][0]
            else:
                # Fallback to first chunk
                return [chunks[0].page_content] if chunks else []
                
        except Exception as e:
            print(f"Semantic search failed: {e}")
            # Fallback to keyword search
            keyword_search = KeywordSearch()
            return keyword_search.search(question, chunks, top_k)
    
    def is_available(self) -> bool:
        return (SEMANTIC_SEARCH_AVAILABLE and 
                self.embeddings_model is not None and
                self.client is not None)