# infrastructure/progress_store.py
"""Simple in-memory progress tracking with size limit"""
from typing import Dict, Optional
from datetime import datetime, timezone
from core.enums import ProcessingStatus, ErrorCode


class ProgressStore:
    """
    Thread-safe in-memory storage for tracking document processing progress.
    
    Stores real-time status updates that clients poll via /processing-status/{doc_id}.
    Automatically removes old entries to prevent memory leaks.
    
    Usage:
        # Start tracking
        progress_store.start(doc_id, "report.pdf")
        
        # Update progress
        progress_store.update(doc_id, ProcessingStatus.EXTRACTING_TEXT, 45, "Processing page 3/10...")
        
        # Mark complete or failed
        progress_store.complete(doc_id)
        progress_store.fail(doc_id, "OCR timeout", ErrorCode.OCR_TIMEOUT)
        
        # Client polls for status
        status = progress_store.get(doc_id)
    
    Cleanup:
        - Removes entries when MAX_ENTRIES (500) reached
        - Keeps newest 250, removes oldest 250
        - Happens automatically on each start() call
    
    Storage Format:
        {
            "doc_id": {
                "filename": str,
                "status": ProcessingStatus,
                "progress_percent": int (0-100),
                "current_step": str,
                "error": Optional[str],
                "error_code": Optional[ErrorCode],
                "_created": datetime
            }
        }
    
    Thread Safety: NOT thread-safe (use in single-threaded async context)
    Persistence: In-memory only (lost on server restart)
    """
    MAX_ENTRIES = 500
    
    def __init__(self):
        self._progress: Dict[str, Dict] = {}
    
    def _cleanup_if_full(self):
        """Remove oldest entries when limit reached"""
        if len(self._progress) <= self.MAX_ENTRIES:
            return
        
        # Keep newest 250, remove oldest 250
        sorted_items = sorted(
            self._progress.items(),
            key=lambda x: x[1].get('_created', datetime.min.replace(tzinfo=timezone.utc))
        )
        to_remove = len(self._progress) - 250
        for doc_id, _ in sorted_items[:to_remove]:
            del self._progress[doc_id]
    
    def start(self, document_id: str, filename: str) -> None:
        self._cleanup_if_full()
        self._progress[document_id] = {
            "filename": filename,
            "status": ProcessingStatus.PENDING,
            "progress_percent": 0,
            "current_step": "Starting...",
            "error": None,
            "error_code": None,
            "_created": datetime.now(timezone.utc)
        }
    
    def update(self, document_id: str, status: ProcessingStatus, 
               progress: int, step: str) -> None:
        if document_id in self._progress:
            self._progress[document_id].update({
                "status": status,
                "progress_percent": progress,
                "current_step": step
            })
    
    def fail(self, document_id: str, error: str, error_code: ErrorCode) -> None:
        if document_id in self._progress:
            self._progress[document_id].update({
                "status": ProcessingStatus.FAILED,
                "error": error,
                "error_code": error_code
            })
    
    def complete(self, document_id: str) -> None:
        if document_id in self._progress:
            self._progress[document_id].update({
                "status": ProcessingStatus.COMPLETED,
                "progress_percent": 100,
                "current_step": "Done!"
            })
    
    def get(self, document_id: str) -> Optional[Dict]:
        return self._progress.get(document_id)
    
    def remove(self, document_id: str) -> None:
        self._progress.pop(document_id, None)


# Global instance
progress_store = ProgressStore()