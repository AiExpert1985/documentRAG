# main.py
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from api.endpoints import router
from services.logger_config import setup_logging
from services.config import APP_TITLE
from database.chat_db import Base, async_engine, delete_file_hash
from services.rag_service import RAGService
from services.llm_service import LLMService

# Setup logging before anything else
setup_logging()

# Global instances that will be set during startup
_rag_service = None
_llm_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the application's lifespan events.
    Creates database tables and service instances on startup.
    Performs cleanup on shutdown.
    """
    # Startup: Create database tables
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    # Instantiate global services
    global _rag_service, _llm_service
    _rag_service = RAGService()
    _llm_service = LLMService()

    yield

    # Shutdown: Clear all resources
    await delete_file_hash()
    await _rag_service.clear_all_documents()

app = FastAPI(title=APP_TITLE, lifespan=lifespan)

# Mount static files.
app.mount("/static", StaticFiles(directory="static"), name="static")

# Include API routes
app.include_router(router, tags=["API Endpoints"])

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)