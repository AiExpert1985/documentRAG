# database/chat_db.py
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, DateTime
from datetime import datetime
import hashlib
import os

DATABASE_URL = "sqlite+aiosqlite:///./chat.db"
DOC_HASHES_FILE = "./document_hashes.txt"

async_engine = create_async_engine(
    DATABASE_URL,
    pool_size=20,
    max_overflow=30
)
AsyncSessionLocal = async_sessionmaker(async_engine, expire_on_commit=False)
Base = declarative_base()

class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, index=True)
    sender = Column(String)
    content = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)

async def get_async_db():
    async with AsyncSessionLocal() as session:
        yield session

def get_file_hash(file_content: bytes) -> str:
    """Generate a hash for a file's content to check for duplicates."""
    return hashlib.sha256(file_content).hexdigest()

def save_file_hash(file_hash: str):
    """Save a file hash to a local file."""
    with open(DOC_HASHES_FILE, "a") as f:
        f.write(file_hash + "\n")

def check_file_hash(file_hash: str) -> bool:
    """Check if a file hash already exists."""
    if not os.path.exists(DOC_HASHES_FILE):
        return False
    with open(DOC_HASHES_FILE, "r") as f:
        for line in f:
            if line.strip() == file_hash:
                return True
    return False