# infrastructure/embedding_services.py
"""Embedding generation implementations"""
import asyncio
import logging
from typing import List, Optional
from sentence_transformers import SentenceTransformer

from core.interfaces import IEmbeddingService
from config import settings

logger = logging.getLogger(settings.LOGGER_NAME) # Use configured logger

class SentenceTransformerEmbedding(IEmbeddingService):
    """
    Sentence transformer embedding service with singleton pattern and offline usage.
    
    Loads the heavy model only once and uses a fallback for initial download.
    """
    
    # Class variable to hold the initialized model instance (Singleton cache)
    _model: Optional[SentenceTransformer] = None 
    
    def __init__(self, model_name: str = "paraphrase-multilingual-mpnet-base-v2"):
        """Initializes the service, loading the heavy model only once."""
        
        # Singleton Logic
        if SentenceTransformerEmbedding._model is None:
            try:
                # CRITICAL FIX: Try loading offline first (no network check/timeout)
                logger.info(f"Attempting to load model {model_name} from local cache...")
                SentenceTransformerEmbedding._model = SentenceTransformer(
                    model_name,
                    local_files_only=True 
                )
                logger.info(f"Successfully loaded {model_name} from local cache.")

            except Exception as e:
                # GRACEFUL FALLBACK: If local load fails, try downloading it (network required)
                logger.warning(
                    f"Model {model_name} not found in cache. Attempting online download. "
                    f"This may take a few minutes. Error: {e}"
                )
                SentenceTransformerEmbedding._model = SentenceTransformer(model_name)
                logger.info(f"Successfully downloaded and loaded {model_name}.")
            
        # Assign the shared model instance to the instance variable
        self.model = SentenceTransformerEmbedding._model
    
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