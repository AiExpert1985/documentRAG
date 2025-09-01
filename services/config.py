# services/config.py
import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """
    Loads configuration from environment variables.
    Create a .env file in the root directory to set these values.
    """
    # Logger configuration
    LOGGER_NAME: str = "rag_system_logger"
    LOG_FILE_PATH: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'rag_system.log')

    # Database and models
    DATABASE_URL: str = "sqlite+aiosqlite:///./chat.db"
    VECTOR_DB_PATH: str = "./vector_db"
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    LLM_MODEL_NAME: str = "llama3.1:8b"
    LLM_BASE_URL: str = "http://localhost:11434"

    # Basic limits
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB
    REQUEST_TIMEOUT: int = 60  # seconds

    # App metadata
    APP_TITLE: str = 'Document RAG System'

    class Config:
        # This allows loading variables from a .env file
        env_file = ".env"
        env_file_encoding = 'utf-8'

# Create a single settings instance to be used across the application
settings = Settings()