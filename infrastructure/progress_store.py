"""Simple in-memory progress tracking with size limit"""
from typing import Dict, Optional
from datetime import datetime
from api.schemas import ProcessingStatus, ErrorCode

class ProgressStore:
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
            key=lambda x: x[1].get('_created', datetime.min)
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
            "_created": datetime.utcnow()
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