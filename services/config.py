# services/config.py
import os

# Logger configuration
LOGGER_NAME = "rag_system_logger"
LOG_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rag_system.log')

# Database and models
VECTOR_DB_PATH = "./vector_db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = "llama3.1:8b"
LLM_BASE_URL = "http://localhost:11434"

# Basic limits
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
REQUEST_TIMEOUT = 30  # seconds

# App metadata
APP_TITLE = 'Document RAG System'