# config.py
from pydantic_settings import BaseSettings
from utils.paths import get_log_file_path

class Settings(BaseSettings):
    """
    Loads configuration from environment variables.
    Create a .env file in the root directory to set these values.
    """
    # Logger configuration
    LOGGER_NAME: str = "rag_system_logger"
    # Set the LOG_FILE_PATH by calling our utility function by default.
    # It can still be overridden by an environment variable if needed.
    LOG_FILE_PATH: str = get_log_file_path()

    # Database and models
    DATABASE_URL: str = "sqlite+aiosqlite:///./chat.db"
    VECTOR_DB_PATH: str = "./vector_db"
    EMBEDDING_MODEL_NAME: str = "paraphrase-multilingual-mpnet-base-v2"
    
    # Basic limits
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB
    REQUEST_TIMEOUT: int = 60  # seconds

    # App metadata
    APP_TITLE: str = 'Document RAG System'
        
    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'

# Create a single settings instance to be used across the application
settings = Settings()