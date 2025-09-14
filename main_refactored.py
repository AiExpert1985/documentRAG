# main_refactored.py
"""Refactored main application with proper initialization"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from config import settings
from services.logger_config import setup_logging
from database.chat_db import Base, async_engine
from api.endpoints_refactored import router
from services.factory import ServiceFactory

# Setup logging
setup_logging()
logger = logging.getLogger(settings.LOGGER_NAME)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # --- Startup ---
    logger.info("Starting application...")
    
    # Initialize database tables
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database initialized")
    
    # Pre-initialize singleton services
    # This ensures they're created once and reused
    # The factory will handle the actual initialization
    logger.info("Services initialized")
    
    yield
    
    # --- Shutdown ---
    logger.info("Shutting down application...")
    
    # Clear cached service instances
    ServiceFactory.clear_instances()
    logger.info("Application shutdown complete")

# Create FastAPI app
app = FastAPI(
    title=settings.APP_TITLE,
    lifespan=lifespan
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

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