# database/chat_db.py
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import Column, Integer, String, DateTime, select, delete
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
import hashlib
import logging

from services.config import LOGGER_NAME

logger = logging.getLogger(LOGGER_NAME)

DATABASE_URL = "sqlite+aiosqlite:///./chat.db"

async_engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = async_sessionmaker(async_engine, expire_on_commit=False)
Base = declarative_base()

class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, index=True)
    sender = Column(String)  # 'user' or 'ai'
    content = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)

class FileHash(Base):
    __tablename__ = "file_hashes"
    id = Column(Integer, primary_key=True, index=True)
    hash_value = Column(String, unique=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)

def get_file_hash(file_content: bytes) -> str:
    return hashlib.sha256(file_content).hexdigest()

async def check_file_hash(file_hash: str) -> bool:
    try:
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(FileHash).where(FileHash.hash_value == file_hash)
            )
            return result.scalar_one_or_none() is not None
    except Exception as e:
        logger.error(f"Error checking file hash: {e}")
        return False

async def save_file_hash(file_hash: str) -> bool:
    try:
        async with AsyncSessionLocal() as session:
            new_hash = FileHash(hash_value=file_hash)
            session.add(new_hash)
            await session.commit()
            return True
    except Exception as e:
        logger.error(f"Error saving file hash: {e}")
        return False

async def delete_file_hash(file_hash: str = None) -> bool:
    try:
        async with AsyncSessionLocal() as session:
            if file_hash:
                await session.execute(
                    delete(FileHash).where(FileHash.hash_value == file_hash)
                )
            else:
                await session.execute(delete(FileHash))
            await session.commit()
            return True
    except Exception as e:
        logger.error(f"Error deleting file hash: {e}")
        return False