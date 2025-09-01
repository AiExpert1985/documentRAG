# main.py
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import chromadb
from chromadb.utils import embedding_functions

from api.endpoints import router
from services.logger_config import setup_logging
from services.config import settings
from database.chat_db import Base, async_engine
from services.llm_service import LLMService
from services.rag_service import RAGService
from services.retrieval_strategies import SemanticRetrieval

# Setup logging as the first step
setup_logging()
logger = logging.getLogger(settings.LOGGER_NAME)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup ---
    logger.info("Application startup...")

    # Initialize Database
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables created.")

    # Initialize ChromaDB Client (singleton)
    chroma_client = chromadb.PersistentClient(path=settings.VECTOR_DB_PATH)
    
    # Initialize Embedding Function (singleton)
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=settings.EMBEDDING_MODEL_NAME
    )
    logger.info(f"Embedding model '{settings.EMBEDDING_MODEL_NAME}' loaded.")
    
    # Initialize LLM Service (singleton)
    app.state.llm_service = LLMService(base_url=settings.LLM_BASE_URL, model=settings.LLM_MODEL_NAME)
    logger.info(f"LLM service configured for model '{settings.LLM_MODEL_NAME}'.")

    # Initialize Retrieval Strategy (singleton)
    semantic_retrieval = SemanticRetrieval(chroma_client, embedding_function)

    # Initialize RAG Service (singleton)
    app.state.rag_service = RAGService(strategy=semantic_retrieval)
    logger.info("RAG service initialized with SemanticRetrieval strategy.")
    
    yield
    
    # --- Shutdown ---
    logger.info("Application shutdown...")

app = FastAPI(title=settings.APP_TITLE, lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)