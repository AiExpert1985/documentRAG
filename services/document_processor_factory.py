# services/document_processor_factory.py
"""Improved factory with proper separation of concerns"""
from typing import Dict, Type, Optional
from core.interfaces import IDocumentProcessor, IPdfToImageConverter
from infrastructure.document_processors import (
    EasyOCRProcessor, 
    PaddleOCRProcessor
)
from config import settings
from infrastructure.pdf_converters import PyMuPDFConverter

class DocumentProcessorFactory:
    """Factory for document processors with strategy selection"""
    
    def __init__(self, pdf_converter_class=None):
        self._ocr_strategies: Dict[str, Type[IDocumentProcessor]] = {
            "easyocr": EasyOCRProcessor,
            "paddleocr": PaddleOCRProcessor
        }
        self.pdf_converter_class = pdf_converter_class or PyMuPDFConverter

    def _get_pdf_processor(self) -> IDocumentProcessor:
        """Gets the OCR-based document processor configured for PDF handling."""
        if settings.PDF_PROCESSING_METHOD == "ocr":
            # --- FIX #2: RENAMED METHOD CALL ---
            return self._create_ocr_processor()
        # Future: elif settings.PDF_PROCESSING_METHOD == "text_extraction":
        #     return TextExtractionProcessor()
        else:
            raise ValueError(f"Unknown PDF processing method: {settings.PDF_PROCESSING_METHOD}")

    # --- FIX #2: RENAMED AND REFACTORED METHOD ---
    def _create_ocr_processor(self) -> IDocumentProcessor:
        """Create OCR processor with selected engine and inject a fresh PDF converter."""
        engine = settings.OCR_ENGINE
        processor_class = self._ocr_strategies.get(engine.lower())
        
        if not processor_class:
            available = ", ".join(self._ocr_strategies.keys())
            raise ValueError(f"Unknown OCR engine: '{engine}'. Available: {available}")
        
        # Concurrency Fix: Create new instance of the converter class
        pdf_converter = self.pdf_converter_class()
        return processor_class(pdf_converter=pdf_converter)
    
    def get_processor(self, file_type: str) -> IDocumentProcessor:
        """Get processor based on file type."""
        # --- FIX #1: REMOVED file_type = file_type.lower() ---
        
        if file_type == "pdf":
            return self._get_pdf_processor()
        # --- FIX #2: USE CONFIG FOR IMAGE EXTENSIONS ---
        elif file_type in settings.IMAGE_EXTENSIONS:
            engine = settings.OCR_ENGINE
            processor_class = self._ocr_strategies.get(engine.lower())
            if not processor_class:
                available = ", ".join(self._ocr_strategies.keys())
                raise ValueError(f"Unknown OCR engine: '{engine}'. Available: {available}")
            return processor_class(pdf_converter=None)
        # ------------------------------------------------
        else:
            raise ValueError(f"Unsupported file type: {file_type}")