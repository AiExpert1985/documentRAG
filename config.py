# config.py
"""Enhanced configuration with new settings"""
from pydantic_settings import BaseSettings
from utils.paths import get_log_file_path

class Settings(BaseSettings):
    """Application configuration"""
    
    # Logger configuration
    LOGGER_NAME: str = "rag_system_logger"
    LOG_FILE_PATH: str = get_log_file_path()
    
    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./chat.db"
    
    # Vector store
    VECTOR_DB_PATH: str = "./vector_db"
    VECTOR_STORE_TYPE: str = "chromadb"  # Can be switched to "weaviate", "pinecone", etc.
    
    # Embedding model
    EMBEDDING_MODEL_NAME: str = "paraphrase-multilingual-mpnet-base-v2"
    
    # Document processing
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB
    SUPPORTED_FILE_TYPES: list = ["pdf"]  # Can be extended to ["pdf", "docx", "txt"]
    
    # API settings
    REQUEST_TIMEOUT: int = 60
    MAX_SEARCH_RESULTS: int = 10
    
    # App metadata
    APP_TITLE: str = "Document RAG System"
    APP_VERSION: str = "2.0.0"
    
    # Performance settings
    USE_CONNECTION_POOLING: bool = True
    MAX_CONNECTIONS: int = 10
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()