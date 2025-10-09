# infrastructure/faiss_store.py
import asyncio
import logging
import json
from typing import List, Optional, Dict, Any
from pathlib import Path

import faiss
import numpy as np

from core.interfaces import IVectorStore, DocumentChunk
from core.domain import ChunkSearchResult
from config import settings

logger = logging.getLogger(settings.LOGGER_NAME)

class FAISSVectorStore(IVectorStore):
    """FAISS implementation of the vector store"""
    
    def __init__(self, index_path: str = settings.VECTOR_DB_PATH):
        self._index: Optional[faiss.IndexFlatL2] = None
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._index_path = Path(index_path) / "faiss.index"
        self._metadata_path = Path(index_path) / "faiss_metadata.json"
        
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        self._load_index()

    def _load_index(self):
        """Loads index and metadata from disk"""
        if self._index_path.exists():
            try:
                self._index = faiss.read_index(str(self._index_path))
                logger.info(f"[FAISS] Loaded index from {self._index_path}")
            except Exception as e:
                logger.warning(f"[FAISS] Failed to load index: {e}. Starting fresh.")
                self._index = None
        
        if self._metadata_path.exists():
            try:
                with open(self._metadata_path, 'r', encoding='utf-8') as f:
                    self._metadata = json.load(f)
                logger.info(f"[FAISS] Loaded {len(self._metadata)} metadata records.")
            except Exception as e:
                logger.warning(f"[FAISS] Failed to load metadata: {e}. Starting fresh.")
                self._metadata = {}

    def _save_index(self):
        """Saves index and metadata to disk"""
        if self._index:
            faiss.write_index(self._index, str(self._index_path))
        
        with open(self._metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self._metadata, f, indent=2, ensure_ascii=False)
        logger.info(f"[FAISS] Saved index and metadata.")

    async def add_chunks(self, chunks: List[DocumentChunk]) -> bool:
        if not chunks:
            return True

        embeddings = np.array([c.embedding for c in chunks], dtype='float32')

        if self._index is None:
            dim = embeddings.shape[1]
            self._index = faiss.IndexFlatL2(dim)
            logger.info(f"[FAISS] Initialized new index with dimension {dim}")

        try:
            await asyncio.to_thread(self._index.add, embeddings) # type: ignore

            for chunk in chunks:
                self._metadata[chunk.id] = {
                    "id": chunk.id,
                    "content": chunk.content,
                    "document_id": chunk.document_id,
                    "metadata": chunk.metadata
                }
            
            self._save_index()
            return True
        except Exception as e:
            logger.error(f"[FAISS] Failed to add chunks: {e}")
            return False

    async def search(self, query_embedding: List[float], top_k: int = 5) -> List[ChunkSearchResult]:
        if not self._index:
            logger.warning("[FAISS] Search called but index is empty.")
            return []

        try:
            query_vector = np.array([query_embedding], dtype='float32')
            distances, indices = await asyncio.to_thread(
                self._index.search, query_vector, top_k
            )  # type: ignore
            
            results = []
            all_chunk_ids = list(self._metadata.keys())
            
            for i in range(top_k):
                index = indices[0][i]
                if index == -1 or index >= len(all_chunk_ids):
                    continue
                
                chunk_id = all_chunk_ids[index]
                metadata = self._metadata.get(chunk_id)
                
                if metadata:
                    chunk = DocumentChunk(
                        id=chunk_id,
                        content=metadata['content'],
                        document_id=metadata['document_id'],
                        metadata=metadata['metadata']
                    )
                    score = 1.0 / (1.0 + distances[0][i])
                    results.append(ChunkSearchResult(chunk=chunk, score=score))
            
            return results
        except Exception as e:
            logger.error(f"[FAISS] Search failed: {e}")
            return []

    async def delete_by_document(self, document_id: str) -> bool:
        """Delete all chunks for a document and rebuild index"""
        if not self._index or not self._metadata:
            return True

        try:
            vectors_to_keep = []
            metadata_to_keep = {}
            all_chunk_ids = list(self._metadata.keys())
            
            # Reconstruct vectors we want to keep
            def reconstruct_keepers():
                keepers = []
                for idx, chunk_id in enumerate(all_chunk_ids):
                    chunk_data = self._metadata[chunk_id]
                    if chunk_data['document_id'] != document_id:
                        # Reconstruct vector from FAISS index (using its sequential index)
                        vector = self._index.reconstruct(idx)  # type: ignore
                        keepers.append((chunk_id, vector, chunk_data))
                return keepers
            
            keepers = await asyncio.to_thread(reconstruct_keepers)
            
            # Rebuild index
            if keepers:
                for chunk_id, vector, chunk_data in keepers:
                    vectors_to_keep.append(vector)
                    metadata_to_keep[chunk_id] = chunk_data
                
                embeddings = np.array(vectors_to_keep, dtype='float32')
                dim = embeddings.shape[1]
                self._index = faiss.IndexFlatL2(dim)
                await asyncio.to_thread(self._index.add, embeddings)  # type: ignore
            else:
                self._index = None
            
            self._metadata = metadata_to_keep
            self._save_index()
            
            logger.info(f"[FAISS] Deleted document {document_id}. "
                       f"Remaining chunks: {len(metadata_to_keep)}")
            return True
            
        except Exception as e:
            logger.error(f"[FAISS] Deletion failed: {e}")
            return False

    async def clear(self) -> bool:
        self._index = None
        self._metadata = {}
        
        try:
            if self._index_path.exists():
                await asyncio.to_thread(self._index_path.unlink)
            if self._metadata_path.exists():
                await asyncio.to_thread(self._metadata_path.unlink)
            return True
        except Exception as e:
            logger.error(f"[FAISS] Clear failed: {e}")
            return False

    async def count(self) -> int:
        return self._index.ntotal if self._index else 0