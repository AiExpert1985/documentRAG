# services/document_processor_factory.py
"""Improved factory with proper separation of concerns"""
from typing import Dict, Type
from core.interfaces import IDocumentProcessor, IPdfToImageConverter
from infrastructure.document_processors import (
    EasyOCRProcessor, 
    PaddleOCRProcessor,
    TesseractProcessor
)
from config import settings
from infrastructure.pdf_converters import PyMuPDFConverter

class DocumentProcessorFactory:
    """Factory for document processors with strategy selection"""
    
    def __init__(self, pdf_converter_class=None):
        self._ocr_strategies: Dict[str, Type[IDocumentProcessor]] = {
            "easyocr": EasyOCRProcessor,
            "paddleocr": PaddleOCRProcessor,
            "tesseract": TesseractProcessor  # Add this
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
        """
        Creates an OCR processor instance with the configured engine strategy.
        
        Retrieves the processor class based on the OCR_ENGINE setting (easyocr or
        paddleocr) and instantiates it with or without a PDF converter depending
        on whether the source file is a PDF or direct image.
        
        Args:
            needs_pdf_converter: If True, includes a PDF-to-image converter.
                            If False, processor expects direct image input.
                            Default is False for image-only processing.
            
        Returns:
            IDocumentProcessor: Configured OCR processor ready for text extraction
            
        Raises:
            ValueError: If configured OCR engine is not found in available strategies
            
        Example:
            For PDF processing

            processor = factory._create_ocr_processor(needs_pdf_converter=True)
            
            For image processing 

            processor = factory._create_ocr_processor(needs_pdf_converter=False)
            
        Note:
            This is a private helper method. External code should use get_processor()
            instead. Creates a fresh PDF converter instance on each call to avoid
            concurrency issues with shared state.
        """
        engine = settings.OCR_ENGINE.lower()
        processor_class = self._ocr_strategies.get(engine)
        
        if not processor_class:
            available = ", ".join(self._ocr_strategies.keys())
            raise ValueError(f"Unknown OCR engine: '{engine}'. Available: {available}")
        
        pdf_converter = self.pdf_converter_class() if needs_pdf_converter else None
        return processor_class(
            pdf_converter=pdf_converter, # type: ignore
            chunk_size=settings.CHUNK_SIZE, # type: ignore
            chunk_overlap=settings.CHUNK_OVERLAP # type: ignore
        )
    
    def get_processor(self, file_type: str) -> IDocumentProcessor:
        """
        Returns the appropriate document processor based on file type.
        
        Selects and instantiates the correct processor implementation using the
        configured OCR engine strategy. For PDFs, includes a PDF-to-image converter.
        For direct images, no converter is needed.
        
        Args:
            file_type: Normalized file extension (e.g., "pdf", "jpg", "png")
                    Should be lowercase without dot (use get_file_extension())
            
        Returns:
            IDocumentProcessor: Configured processor instance ready to extract text
            
        Raises:
            ValueError: If file type is unsupported or OCR engine is unknown
            
        Note:
            Creates a new processor instance for each call to ensure stateless
            processing. The underlying OCR engine (EasyOCR/PaddleOCR reader) is
            shared across instances for performance.
        """
        if file_type == "pdf":
            return self._get_pdf_processor()
        elif file_type in settings.IMAGE_EXTENSIONS:
            return self._create_ocr_processor(needs_pdf_converter=False)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")