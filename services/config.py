# services/config.py
"""
Central configuration for the RAG system.
Stores global constants and settings in one place.
"""
import os

# Logger configuration
LOGGER_NAME = "rag_system_logger"
LOG_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rag_system.log')

# Database and models
VECTOR_DB_PATH = "./vector_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama3.1:8b"
LLM_BASE_URL = "http://localhost:11434"

# titles
APP_TITLE = 'Document RAG System'