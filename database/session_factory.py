# database/session_factory.py
"""Centralized session management for background tasks"""
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession
from database.models import AsyncSessionLocal

@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Create a new database session with proper cleanup.
    
    Used by background processors where request-scoped sessions are unavailable.
    Ensures proper rollback on errors and explicit closure.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()