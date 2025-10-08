# config.py
"""Enhanced configuration with new settings"""
from typing import List
from pydantic_settings import BaseSettings
from utils.paths import get_log_file_path, get_project_root

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

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()