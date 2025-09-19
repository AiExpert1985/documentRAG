# database/chat_db.py
import logging
import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, Integer, String, select
from sqlalchemy.ext.asyncio import (AsyncSession, async_sessionmaker,
                                    create_async_engine)
from sqlalchemy.orm import declarative_base

from config import settings

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
    filename = Column(String, nullable=False) # The original filename
    file_hash = Column(String, unique=True, index=True, nullable=False)
    # CHANGED: Add a column to store the secure filename on disk
    stored_filename = Column(String, nullable=False, unique=True)
    timestamp = Column(DateTime, default=datetime.utcnow)

