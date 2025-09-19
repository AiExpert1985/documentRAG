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
    
    async def create(self, document_id: str, filename: str, file_hash: str, stored_filename: str) -> DomainDocument:
        db_doc = DBDocument(
            id=document_id,
            filename=filename,
            file_hash=file_hash,
            stored_filename=stored_filename
        )
        self.session.add(db_doc)
        await self.session.commit()
        await self.session.refresh(db_doc)
        
        return self._to_domain(db_doc)

    
    def _to_domain(self, db_doc: DBDocument) -> Optional[DomainDocument]:
        if not db_doc:
            return None
        return DomainDocument(
            id=db_doc.id,
            filename=db_doc.filename,
            file_hash=db_doc.file_hash,
            metadata={
                "timestamp": db_doc.timestamp.isoformat(),
                "stored_filename": db_doc.stored_filename
            }
        )
    


    async def get_by_id(self, document_id: str) -> Optional[DomainDocument]:
        db_doc = await self.session.get(DBDocument, document_id)
        return self._to_domain(db_doc)

    async def get_by_hash(self, file_hash: str) -> Optional[DomainDocument]:
        result = await self.session.execute(
            select(DBDocument).where(DBDocument.file_hash == file_hash)
        )
        db_doc = result.scalar_one_or_none()
        return self._to_domain(db_doc)

    async def list_all(self) -> List[DomainDocument]:
        result = await self.session.execute(
            select(DBDocument).order_by(DBDocument.timestamp.desc())
        )
        docs = result.scalars().all()
        return [self._to_domain(doc) for doc in docs]
    
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