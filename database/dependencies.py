# database/dependencies.py
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession
from database.models import AsyncSessionLocal

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Provide database session for FastAPI dependency injection"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()