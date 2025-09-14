# infrastructure/embedding_services.py
"""Embedding generation implementations"""
import asyncio
from typing import List
from sentence_transformers import SentenceTransformer

from core.interfaces import IEmbeddingService

class SentenceTransformerEmbedding(IEmbeddingService):
    """Sentence transformer embedding service"""
    
    def __init__(self, model_name: str = "paraphrase-multilingual-mpnet-base-v2"):
        self.model = SentenceTransformer(model_name)
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        embeddings = await asyncio.to_thread(
            self.model.encode,
            texts,
            convert_to_tensor=False
        )
        return embeddings.tolist()
    
    async def generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for query"""
        embedding = await asyncio.to_thread(
            self.model.encode,
            query,
            convert_to_tensor=False
        )
        return embedding.tolist()