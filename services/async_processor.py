# services/async_processor.py
"""Simple async document processor using threading (good for MVP)"""
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Coroutine

logger = logging.getLogger(__name__)

class AsyncDocumentProcessor:
    """Processes documents in background threads"""
    
    def __init__(self, max_workers: int = 3):
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
    
    def submit_task(self, coro: Coroutine) -> None:
        """Submit an async task to run in background"""
        def run_async():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(coro)
                loop.close()
            except Exception as e:
                logger.exception(f"Background task failed: {e}")
        
        self._executor.submit(run_async)
    
    def shutdown(self):
        """Clean shutdown - wait for running tasks"""
        self._executor.shutdown(wait=True)

# Global instance
async_processor = AsyncDocumentProcessor(max_workers=3)