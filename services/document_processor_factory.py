# services/document_processor_factory.py
"""Improved factory with proper separation of concerns"""
from typing import Dict, Type, Optional
from core.interfaces import IDocumentProcessor
from infrastructure.document_processors import (
    PDFProcessor, 
    EasyOCRProcessor, 
    PaddleOCRProcessor
)
from infrastructure.pdf_converters import PyMuPDFConverter
from config import settings

class DocumentProcessorFactory:
    """Factory for document processors with strategy selection"""
    
    def __init__(self):
        self._pdf_strategies: Dict[str, Type[IDocumentProcessor]] = {
            "unstructured": PDFProcessor,
            "easyocr": EasyOCRProcessor,
            "paddleocr": PaddleOCRProcessor
        }
        self.pdf_converter = PyMuPDFConverter()
    
    def get_processor(
        self, 
        file_type: str, 
        strategy: Optional[str] = None
    ) -> IDocumentProcessor:
        """Get appropriate processor based on file type and strategy."""
        file_type = file_type.lower()
        
        if file_type == "pdf":
            return self._get_pdf_processor(strategy)
        # Future: elif file_type == "docx": ...
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    
    def _get_pdf_processor(self, strategy: Optional[str]) -> IDocumentProcessor:
        """Get PDF processor with strategy selection"""
        final_strategy = strategy or settings.DEFAULT_OCR_STRATEGY
        
        processor_class = self._pdf_strategies.get(final_strategy.lower())
        if not processor_class:
            available = ", ".join(self._pdf_strategies.keys())
            raise ValueError(
                f"Unknown PDF processing strategy: '{final_strategy}'. "
                f"Available: {available}"
            )
        
        if final_strategy.lower() in ["easyocr", "paddleocr"]:
            return processor_class(pdf_converter=self.pdf_converter)
        else:
            return processor_class()
            
    def detect_pdf_strategy(self, file_path: str) -> str:
        """
        Auto-detect if PDF needs OCR or can use direct extraction.
        Returns the name of the best strategy.
        """
        try:
            import fitz
            doc = fitz.open(file_path)
            
            # Check first few pages for significant text content
            total_text_len = 0
            pages_to_check = min(3, doc.page_count)
            for i in range(pages_to_check):
                page = doc[i]
                total_text_len += len(page.get_text().strip())
            
            doc.close()
            
            # If average text per page is very low, it's likely a scanned PDF
            avg_text_len = total_text_len / pages_to_check if pages_to_check > 0 else 0
            if avg_text_len < 100:
                return settings.PREFERRED_OCR_ENGINE # e.g., "easyocr"
            
            return "unstructured"
            
        except Exception:
            # Default to unstructured on any error during detection
            return "unstructured"