# services/document_processor_factory.py
"""Improved factory with proper separation of concerns"""
from typing import Dict, Type
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
        """Gets the document processor configured for PDF handling."""
        if settings.PDF_PROCESSING_METHOD == "ocr":
            return self._create_ocr_processor(needs_pdf_converter=True)
        # Future: elif settings.PDF_PROCESSING_METHOD == "text_extraction":
        #     return TextExtractionProcessor()
        else:
            raise ValueError(f"Unknown PDF processing method: {settings.PDF_PROCESSING_METHOD}")

    def _create_ocr_processor(self, needs_pdf_converter: bool = False) -> IDocumentProcessor:
        """Create OCR processor with selected engine."""
        engine = settings.OCR_ENGINE
        processor_class = self._ocr_strategies.get(engine.lower())
        
        if not processor_class:
            available = ", ".join(self._ocr_strategies.keys())
            raise ValueError(f"Unknown OCR engine: '{engine}'. Available: {available}")
        
        pdf_converter = self.pdf_converter_class() if needs_pdf_converter else None
        return processor_class(pdf_converter=pdf_converter)
    
    def get_processor(self, file_type: str) -> IDocumentProcessor:
        """Get processor based on file type."""
        if file_type == "pdf":
            return self._get_pdf_processor()
        elif file_type in settings.IMAGE_EXTENSIONS:
            return self._create_ocr_processor(needs_pdf_converter=False)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")