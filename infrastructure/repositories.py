# infrastructure/repositories.py
"""Database repository implementations"""
import uuid
from typing import List, Optional, Dict
from datetime import datetime
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from core.interfaces import IDocumentRepository, IMessageRepository, Document as DomainDocument
from database.chat_db import Document as DBDocument, Message

class SQLDocumentRepository(IDocumentRepository):
    """SQLAlchemy document repository"""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def create(self, filename: str, file_hash: str) -> DomainDocument:
        """Create document record"""
        doc_id = str(uuid.uuid4())
        db_doc = DBDocument(
            id=doc_id,
            filename=filename,
            file_hash=file_hash
        )
        self.session.add(db_doc)
        await self.session.commit()
        await self.session.refresh(db_doc)
        
        return DomainDocument(
            id=db_doc.id,
            filename=db_doc.filename,
            file_hash=db_doc.file_hash,
            metadata={"timestamp": db_doc.timestamp.isoformat()}
        )
    
    async def get_by_id(self, document_id: str) -> Optional[DomainDocument]:
        """Get document by ID"""
        db_doc = await self.session.get(DBDocument, document_id)
        if not db_doc:
            return None
        
        return DomainDocument(
            id=db_doc.id,
            filename=db_doc.filename,
            file_hash=db_doc.file_hash,
            metadata={"timestamp": db_doc.timestamp.isoformat()}
        )
    
    async def get_by_hash(self, file_hash: str) -> Optional[DomainDocument]:
        """Check if document exists by hash"""
        result = await self.session.execute(
            select(DBDocument).where(DBDocument.file_hash == file_hash)
        )
        db_doc = result.scalar_one_or_none()
        
        if not db_doc:
            return None
        
        return DomainDocument(
            id=db_doc.id,
            filename=db_doc.filename,
            file_hash=db_doc.file_hash,
            metadata={"timestamp": db_doc.timestamp.isoformat()}
        )
    
    async def list_all(self) -> List[DomainDocument]:
        """List all documents"""
        result = await self.session.execute(
            select(DBDocument).order_by(DBDocument.timestamp.desc())
        )
        docs = result.scalars().all()
        
        return [
            DomainDocument(
                id=doc.id,
                filename=doc.filename,
                file_hash=doc.file_hash,
                metadata={"timestamp": doc.timestamp.isoformat()}
            )
            for doc in docs
        ]
    
    async def delete(self, document_id: str) -> bool:
        """Delete document"""
        doc = await self.session.get(DBDocument, document_id)
        if not doc:
            return False
        
        await self.session.delete(doc)
        await self.session.commit()
        return True
    
    async def delete_all(self) -> bool:
        """Delete all documents"""
        await self.session.execute(delete(DBDocument))
        await self.session.commit()
        return True

class SQLMessageRepository(IMessageRepository):
    """SQLAlchemy message repository"""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def save_search(self, query: str, results_count: int) -> None:
        """Save search query"""
        message = Message(
            sender="search",
            content=f"Query: {query} | Results: {results_count}"
        )
        self.session.add(message)
        await self.session.commit()
    
    async def get_search_history(self, limit: int = 10) -> List[Dict]:
        """Get search history"""
        result = await self.session.execute(
            select(Message)
            .where(Message.sender == "search")
            .order_by(Message.timestamp.desc())
            .limit(limit)
        )
        messages = result.scalars().all()
        
        return [
            {
                "query": msg.content.split(" | ")[0].replace("Query: ", ""),
                "timestamp": msg.timestamp.isoformat()
            }
            for msg in reversed(messages)
        ]
    
    async def clear_history(self) -> bool:
        """Clear search history"""
        await self.session.execute(
            delete(Message).where(Message.sender == "search")
        )
        await self.session.commit()
        return True