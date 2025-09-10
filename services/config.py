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
    
    # Database and models
    DATABASE_URL: str = "sqlite+aiosqlite:///./chat.db"
    VECTOR_DB_PATH: str = "./vector_db"
    EMBEDDING_MODEL_NAME: str = "paraphrase-multilingual-mpnet-base-v2"
    LLM_MODEL_NAME: str = "llama3.1:8b"
    LLM_BASE_URL: str = "http://localhost:11434"
    CHAT_CONTEXT_LIMIT: int = 4

    # Basic limits
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB
    REQUEST_TIMEOUT: int = 60  # seconds

    # App metadata
    APP_TITLE: str = 'Document RAG System'

    # The log file path needs to be a computed property to be OS-agnostic
    @property
    def log_file_path(self):
        # This assumes the config.py file is at services/config.py
        # and the project root is two levels up.
        base_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(base_dir, '..', '..'))
        
        log_dir = os.path.join(project_root, 'log')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        return os.path.join(log_dir, 'rag_system.log')
        
    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'

# Create a single settings instance to be used across the application
settings = Settings()