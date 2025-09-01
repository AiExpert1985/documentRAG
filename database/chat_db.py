# database/chat_db.py
import hashlib
import logging
import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, Integer, String, select
from sqlalchemy.ext.asyncio import (AsyncSession, async_sessionmaker,
                                    create_async_engine)
from sqlalchemy.orm import declarative_base

from services.config import settings

logger = logging.getLogger(settings.LOGGER_NAME)

# Setup SQLAlchemy async engine and session maker
async_engine = create_async_engine(settings.DATABASE_URL, echo=False)
AsyncSessionLocal = async_sessionmaker(async_engine, expire_on_commit=False)
Base = declarative_base()

# --- SQLAlchemy Models ---

class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, index=True)
    sender = Column(String, nullable=False)  # 'user' or 'ai'
    content = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

class Document(Base):
    __tablename__ = "documents"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    filename = Column(String, nullable=False)
    file_hash = Column(String, unique=True, index=True, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

# --- Utility Functions ---

def get_file_hash(file_content: bytes) -> str:
    """Calculates the SHA256 hash of file content."""
    return hashlib.sha256(file_content).hexdigest()

async def check_file_hash(db: AsyncSession, file_hash: str) -> bool:
    """
    Checks if a document with the given file hash already exists in the database.
    """
    try:
        result = await db.execute(
            select(Document).where(Document.file_hash == file_hash)
        )
        return result.scalar_one_or_none() is not None
    except Exception as e:
        logger.error(f"Error checking file hash '{file_hash}': {e}", exc_info=True)
        # Fail safe: assume hash exists to prevent re-processing on error
        return True