# services/document_processor_factory.py
"""Improved factory with proper separation of concerns"""
from typing import Dict, Type, Optional
from core.interfaces import IDocumentProcessor
from infrastructure.document_processors import (
    EasyOCRProcessor, 
    PaddleOCRProcessor
)
from config import settings
from infrastructure.pdf_converters import PyMuPDFConverter

class DocumentProcessorFactory:
    """Factory for document processors with strategy selection"""
    
    def __init__(self, pdf_converter_class=None):  # ✅ CHANGED: Optional parameter
        self._ocr_strategies: Dict[str, Type[IDocumentProcessor]] = {
            "easyocr": EasyOCRProcessor,
            "paddleocr": PaddleOCRProcessor
        }
        # ✅ CHANGED: Use injected class or default to PyMuPDFConverter
        self.pdf_converter_class = pdf_converter_class or PyMuPDFConverter

    def _get_pdf_processor(self) -> IDocumentProcessor:
        """Get PDF processor based on processing method."""
        if settings.PDF_PROCESSING_METHOD == "ocr":
            return self._get_ocr_processor()
        else:
            raise ValueError(f"Unknown PDF processing method: {settings.PDF_PROCESSING_METHOD}")

    def _get_ocr_processor(self) -> IDocumentProcessor:
        """Get OCR processor based on engine choice."""
        engine = settings.OCR_ENGINE
        processor_class = self._ocr_strategies.get(engine.lower())
        
        if not processor_class:
            available = ", ".join(self._ocr_strategies.keys())
            raise ValueError(f"Unknown OCR engine: '{engine}'. Available: {available}")
        
        # ✅ CHANGED: Create new instance from injected/default class
        pdf_converter = self.pdf_converter_class()
        return processor_class(pdf_converter=pdf_converter)
        

    def get_processor(self, file_type: str) -> IDocumentProcessor:
        """Get processor based on file type."""
        file_type = file_type.lower()
        
        if file_type == "pdf":
            return self._get_pdf_processor()
        elif file_type in ["jpg", "jpeg", "png"]:
            engine = settings.OCR_ENGINE
            processor_class = self._ocr_strategies.get(engine.lower())
            if not processor_class:
                available = ", ".join(self._ocr_strategies.keys())
                raise ValueError(f"Unknown OCR engine: '{engine}'. Available: {available}")
            return processor_class(pdf_converter=None)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")