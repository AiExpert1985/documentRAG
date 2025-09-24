# infrastructure/repositories_imp.py
"""Database repository implementations"""
import json
import logging
from typing import List, Optional, Dict
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from core.interfaces import IDocumentRepository, IMessageRepository, Document as DomainDocument, SearchResult
from database.database import Document as DBDocument, Message

logger = logging.getLogger(__name__)

class SQLDocumentRepository(IDocumentRepository):
    def __init__(self, session: AsyncSession):
        self.session = session
    
    def _to_domain(self, db_doc: DBDocument) -> Optional[DomainDocument]:
        return DomainDocument(
            id=db_doc.id,
            filename=db_doc.filename,
            file_hash=db_doc.file_hash,
            metadata={
                "timestamp": db_doc.timestamp.isoformat(),
                "stored_filename": db_doc.stored_filename
            }
        ) if db_doc else None
    
    async def create(self, document_id: str, filename: str, file_hash: str, stored_filename: str) -> DomainDocument:
        db_doc = DBDocument(
            id=document_id, filename=filename, 
            file_hash=file_hash, stored_filename=stored_filename
        )
        self.session.add(db_doc)
        await self.session.commit()
        await self.session.refresh(db_doc)
        return self._to_domain(db_doc)

    async def get_by_id(self, document_id: str) -> Optional[DomainDocument]:
        db_doc = await self.session.get(DBDocument, document_id)
        return self._to_domain(db_doc)

    async def get_by_hash(self, file_hash: str) -> Optional[DomainDocument]:
        result = await self.session.execute(
            select(DBDocument).where(DBDocument.file_hash == file_hash)
        )
        return self._to_domain(result.scalar_one_or_none())

    async def list_all(self) -> List[DomainDocument]:
        result = await self.session.execute(
            select(DBDocument).order_by(DBDocument.timestamp.desc())
        )
        return [self._to_domain(doc) for doc in result.scalars().all()]
    
    async def delete(self, document_id: str) -> bool:
        doc = await self.session.get(DBDocument, document_id)
        if not doc:
            return False
        await self.session.delete(doc)
        await self.session.commit()
        return True
    
    async def delete_all(self) -> bool:
        try:
            await self.session.execute(delete(DBDocument))
            await self.session.commit()
            return True
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Failed to clear documents: {e}")
            return False

class SQLMessageRepository(IMessageRepository):
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def save_search(self, query: str, results_count: int) -> None:
        message = Message(sender="search", content=f"Query: {query} | Results: {results_count}")
        self.session.add(message)
        await self.session.commit()
    
    async def save_search_results(self, query: str, results: List[SearchResult]) -> None:
        """Save query and results using same structure as live chat"""
        # Save user question
        user_msg = Message(sender="user", content=query)
        self.session.add(user_msg)
        
        if not results:
            # Save "no results" message
            ai_msg = Message(sender="ai", content="No relevant information found.")
            self.session.add(ai_msg)
        else:
            # Save each result as separate AI message (same as live chat)
            for result in results:
                # Store exactly what frontend expects
                result_json = json.dumps({
                    "document_name": result.chunk.metadata.get('document_name'),
                    "page_number": result.chunk.metadata.get('page'),
                    "content_snippet": result.chunk.content[:300] + "..." if len(result.chunk.content) > 300 else result.chunk.content,
                    "document_id": result.chunk.document_id,
                    "download_url": f"http://127.0.0.1:8000/download/{result.chunk.document_id}"
                })
                
                ai_msg = Message(sender="ai_result", content=result_json)
                self.session.add(ai_msg)
        
        await self.session.commit()

    async def get_search_history(self, limit: int = 10) -> List[Dict]:
        """Get complete chat history (both questions and answers)"""
        result = await self.session.execute(
            select(Message)
            .where(Message.sender.in_(["user", "ai", "ai_result"]))
            .order_by(Message.timestamp.asc())
            .limit(limit)
        )
        messages = result.scalars().all()
        
        return [
            {
                "sender": msg.sender,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat()
            }
            for msg in messages
        ]
    
    async def clear_history(self) -> bool:
        await self.session.execute(delete(Message).where(Message.sender.in_(["user", "ai", "ai_result"])))
        await self.session.commit()
        return True