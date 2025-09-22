# main_refactored.py
"""Refactored main application with proper initialization"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI

# --- ADD THESE IMPORTS ---
from fastapi.middleware.cors import CORSMiddleware
# -------------------------

from config import settings
from services.logger_config import setup_logging
from database.database import Base, async_engine
from api.endpoints import router

# Setup logging
setup_logging()
logger = logging.getLogger(settings.LOGGER_NAME)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("Starting application...")

    #✅
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database initialized")
    
    logger.info("Services initialized")
    yield
    
    logger.info("Application shutdown complete")

# Create FastAPI app
app = FastAPI(
    title=settings.APP_TITLE,
    lifespan=lifespan
)

# --- ADD THIS CORS MIDDLEWARE SECTION ---
# This allows your Flutter app (and any other client) to make requests to your API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)
# ------------------------------------

# Include API routes
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )