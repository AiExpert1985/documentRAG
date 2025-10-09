# config.py
"""Enhanced configuration with new settings"""
from typing import List
from pydantic_settings import BaseSettings
from utils.common import get_log_file_path, get_project_root

class Settings(BaseSettings):
    """Application configuration"""
    
    # Logger configuration
    LOGGER_NAME: str = "rag_system_logger"
    LOG_FILE_PATH: str = get_log_file_path()
    
    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./chat.db"
    DB_POOL_SIZE: int = 5
    DB_MAX_OVERFLOW: int = 10
    DB_POOL_RECYCLE: int = 3600  # Recycle connections after 1 hour
    
    # Vector store
    VECTOR_DB_PATH: str = "./vector_db"
    VECTOR_STORE_TYPE: str = "chromadb"
    
    # Embedding model
    EMBEDDING_MODEL_NAME: str = "paraphrase-multilingual-mpnet-base-v2"
    
    # Document processing
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    MAX_FILE_SIZE: int = 50 * 1024 * 1024
    UPLOADS_DIR: str = f"{get_project_root()}/uploads"
    
    # PDF processing method selection
    PDF_PROCESSING_METHOD: str = "ocr"
    OCR_ENGINE: str = "easyocr" # Options: tesseract, easyocr, paddleocr
    OCR_DPI: int = 300
    OCR_LANGUAGES: List[str] = ["ar", "en"]
    
    # File type categorization
    IMAGE_EXTENSIONS: List[str] = ["jpg", "jpeg", "png"]
    DOCUMENT_EXTENSIONS: List[str] = ["pdf"]
    
    @property
    def ALLOWED_FILE_EXTENSIONS(self) -> List[str]:
        return self.IMAGE_EXTENSIONS + self.DOCUMENT_EXTENSIONS
    
    # API/RAG Configuration
    OCR_TIMEOUT_SECONDS: float = 300.0
    SNIPPET_LENGTH: int = 300
    DEFAULT_SEARCH_RESULTS: int = 5
    
    # ============= NEW: Search Quality Controls =============
    # These settings improve search relevance by filtering weak matches
    
    SEARCH_SCORE_THRESHOLD: float = 0.70
    """
    Minimum similarity score (0.0-1.0) for results to be returned.
    - 0.60-0.65: Lenient (more results, some weak matches)
    - 0.70-0.75: Balanced (recommended for most use cases)
    - 0.80-0.85: Strict (only very relevant matches)
    
    This threshold works consistently across FAISS and ChromaDB because
    embeddings are L2-normalized and scores are unified to [0,1] scale.
    """
    
    SEARCH_CANDIDATE_K: int = 15
    """
    Number of candidates to fetch before filtering (overfetch strategy).
    Should be 2-3x larger than DEFAULT_SEARCH_RESULTS.
    
    Why: Ensures we find strong matches even if top-5 raw results are mediocre.
    Example: Fetch 15 candidates, filter by threshold, keep best 5.
    """
    # ============= END: Search Quality Controls =============
    
    # API settings
    REQUEST_TIMEOUT: int = 60
    MAX_SEARCH_RESULTS: int = 10  # Legacy setting
    
    # App metadata
    APP_TITLE: str = "Document RAG System"
    APP_VERSION: str = "2.0.0"
    
    # Performance
    USE_CONNECTION_POOLING: bool = True
    MAX_CONNECTIONS: int = 10

    # Document processing
    ARABIC_BIDI_WRAP_FOR_DISPLAY: bool = True  # Add RTL markers for display


    # Search quality controls
    SEARCH_CANDIDATE_K: int = 40
    SEARCH_SCORE_THRESHOLD: float = 0.72
    
    LEXICAL_GATE_ENABLED: bool = True
    LEXICAL_MIN_KEYWORDS: int = 1
    
    RERANK_ENABLED: bool = True
    RERANK_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RERANK_SCORE_THRESHOLD: float = 0.50
    RERANK_TOP_K: int = 15

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()