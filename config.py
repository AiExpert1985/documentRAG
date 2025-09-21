# config.py
"""Enhanced configuration with new settings"""
from typing import List
from pydantic_settings import BaseSettings
from utils.paths import get_log_file_path
from utils.paths import get_project_root

class Settings(BaseSettings):
    """Application configuration"""
    
    # Logger configuration
    LOGGER_NAME: str = "rag_system_logger"
    LOG_FILE_PATH: str = get_log_file_path()
    
    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./chat.db"
    
    # Vector store
    VECTOR_DB_PATH: str = "./vector_db"
    VECTOR_STORE_TYPE: str = "chromadb"
    
    # Embedding model
    EMBEDDING_MODEL_NAME: str = "paraphrase-multilingual-mpnet-base-v2"
    
    # Document processing
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB
    UPLOADS_DIR: str = f"{get_project_root()}/uploads"
    
    # Document processing strategy configuration
    DEFAULT_OCR_STRATEGY: str = "unstructured"
    PREFERRED_OCR_ENGINE: str = "easyocr"
    AUTO_DETECT_STRATEGY: bool = True  # Let system choose best method
    OCR_DPI: int = 300
    OCR_LANGUAGES: List[str] = ["ar", "en"]

    # File type support
    ALLOWED_FILE_EXTENSIONS: List[str] = ["pdf", "jpg", "jpeg", "png", "docx", "doc", "txt"]

    # API settings
    REQUEST_TIMEOUT: int = 60
    MAX_SEARCH_RESULTS: int = 10
    
    # App metadata
    APP_TITLE: str = "Document RAG System"
    APP_VERSION: str = "2.0.0"
    
    # --- ADDED MISSING SETTINGS ---
    # Performance
    USE_CONNECTION_POOLING: bool = True
    MAX_CONNECTIONS: int = 10
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()