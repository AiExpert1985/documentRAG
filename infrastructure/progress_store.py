# infrastructure/progress_store.py
"""Simple in-memory progress tracking (good for MVP, swap to Redis later)"""
from typing import Dict, Optional
from api.schemas import ProcessingStatus, ErrorCode

class ProgressStore:
    """Thread-safe progress tracking for document processing"""
    
    def __init__(self):
        self._progress: Dict[str, Dict] = {}
    
    def start(self, document_id: str, filename: str) -> None:
        """Initialize progress tracking for a document"""
        self._progress[document_id] = {
            "filename": filename,
            "status": ProcessingStatus.PENDING,
            "progress_percent": 0,
            "current_step": "Starting...",
            "error": None,
            "error_code": None
        }
    
    def update(self, document_id: str, status: ProcessingStatus, 
               progress: int, step: str) -> None:
        """Update processing progress"""
        if document_id in self._progress:
            self._progress[document_id].update({
                "status": status,
                "progress_percent": progress,
                "current_step": step
            })
    
    def fail(self, document_id: str, error: str, 
             error_code: ErrorCode) -> None:
        """Mark processing as failed"""
        if document_id in self._progress:
            self._progress[document_id].update({
                "status": ProcessingStatus.FAILED,
                "error": error,
                "error_code": error_code
            })
    
    def complete(self, document_id: str) -> None:
        """Mark processing as completed"""
        if document_id in self._progress:
            self._progress[document_id].update({
                "status": ProcessingStatus.COMPLETED,
                "progress_percent": 100,
                "current_step": "Done!"
            })
    
    def get(self, document_id: str) -> Optional[Dict]:
        """Get progress for a document"""
        return self._progress.get(document_id)
    
    def remove(self, document_id: str) -> None:
        """Clean up after completion (call after 5 minutes)"""
        self._progress.pop(document_id, None)

# Global instance (swap to Redis later if needed)
progress_store = ProgressStore()