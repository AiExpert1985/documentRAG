# infrastructure/document_processors.py
"""Document processing implementations"""
from abc import abstractmethod
import asyncio
import uuid
from typing import Any, Callable, List, Optional
from PIL import Image
import io
import logging

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LangchainDocument

from core.interfaces import IDocumentProcessor, DocumentChunk, IPdfToImageConverter
from api.schemas import DocumentProcessingError, ErrorCode  # FIX: Added import
from config import settings
from utils.arabic_segmenter import segment_arabic

logger = logging.getLogger(settings.LOGGER_NAME)

class BaseOCRProcessor(IDocumentProcessor):
    def __init__(self, pdf_converter: Optional[IPdfToImageConverter] = None, 
                 chunk_size: int = 1000, chunk_overlap: int = 200):
        self.pdf_converter = pdf_converter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ":", "؛", "؟", ".", "!", "،", " " , ""]
        )

    @abstractmethod
    async def _extract_text_from_image(self, image: Image.Image) -> str:
        """
        Extract text from single image using OCR engine.
        Returns empty string if no text found. Runs with timeout protection.
        """
        pass

    async def process(
        self, 
        file_path: str, 
        file_type: str,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[DocumentChunk]:
        if file_type == 'pdf':
            if not self.pdf_converter:
                raise DocumentProcessingError(
                    "PDF converter required for PDF processing",
                    ErrorCode.PROCESSING_FAILED
                )
            images = await asyncio.to_thread(
                self.pdf_converter.convert, file_path, dpi=settings.OCR_DPI
            )
        elif file_type in settings.IMAGE_EXTENSIONS:
            with Image.open(file_path) as img:
                images = [img.copy()]
        else:
            raise DocumentProcessingError(
                f"Unsupported file type: {file_type}",
                ErrorCode.INVALID_FORMAT
            )

        total_pages = len(images)
        docs: List[LangchainDocument] = []
        
        for i, image in enumerate(images):
            # Report progress before processing page
            if progress_callback:
                progress_callback(i + 1, total_pages)
            
            try:
                text = await asyncio.wait_for(
                    self._extract_text_from_image(image),
                    timeout=settings.OCR_TIMEOUT_SECONDS
                )
            except asyncio.TimeoutError:
                raise DocumentProcessingError(
                    f"OCR timeout on page {i+1} after {settings.OCR_TIMEOUT_SECONDS}s",
                    ErrorCode.OCR_TIMEOUT
                )
            
            if text.strip():
                for segment in segment_arabic(text):
                    docs.append(LangchainDocument(
                        page_content=segment,
                        metadata={"page": i + 1, "source": file_path}
                    ))

        if not docs:
            raise DocumentProcessingError(
                "No text extracted from document",
                ErrorCode.NO_TEXT_FOUND
            )
        
        split_docs = self.text_splitter.split_documents(docs)

        # ✅ OPTIONAL: Merge tiny chunks
        MIN_LEN = 140
        merged = []
        for d in split_docs:
            if merged and len(merged[-1].page_content) < MIN_LEN:
                merged[-1].page_content += "\n" + d.page_content
            else:
                merged.append(d)
        split_docs = merged

        logger.info(f"Processed {len(docs)} pages into {len(split_docs)} chunks")

        return [
            DocumentChunk(
                id=f"{uuid.uuid4()}_{i}",
                content=doc.page_content,
                document_id="",
                metadata={
                    "page": doc.metadata.get("page", -1),
                    "source": doc.metadata.get("source", "")
                }
            )
            for i, doc in enumerate(split_docs)
        ]

class EasyOCRProcessor(BaseOCRProcessor):
    reader: Optional[Any] = None

    def __init__(self, pdf_converter: Optional[IPdfToImageConverter] = None, **kwargs):
        super().__init__(pdf_converter, **kwargs)
        if EasyOCRProcessor.reader is None:
            try:
                import easyocr
                EasyOCRProcessor.reader = easyocr.Reader(settings.OCR_LANGUAGES)
            except ImportError:
                raise ImportError("EasyOCR not installed. Run: pip install easyocr")

    async def _extract_text_from_image(self, image: Image.Image) -> str:
        assert self.reader is not None, "EasyOCR reader not initialized"
        logger.info('Using EasyOCRProcessor')
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='PNG')
        # NOTE: This returns strong structural breaks via \n\n
        result = await asyncio.to_thread(
            self.reader.readtext, img_bytes.getvalue(), detail=0, paragraph=True
        )
        return "\n\n".join(result) if result else ""

class PaddleOCRProcessor(BaseOCRProcessor):
    ocr_engine: Optional[Any] = None
            
    async def _extract_text_from_image(self, image: Image.Image) -> str:
        assert self.ocr_engine is not None, "PaddleOCR engine not initialized"
        logger.info('Using PaddleOCRProcessor')
        import numpy as np
        
        img_array = np.array(image)
        result = await asyncio.to_thread(self.ocr_engine.ocr, img_array)
        
        if not result or result[0] is None:
            return ""
        
        try:
            # FIX: Ensure PaddleOCR output is joined with strong separator
            text = "\n\n".join(line[1][0] for line in result[0]) 
            return text
        except (IndexError, TypeError):
            return ""

class TesseractProcessor(BaseOCRProcessor):
    def __init__(self, pdf_converter: Optional[IPdfToImageConverter] = None, **kwargs):
        super().__init__(pdf_converter, **kwargs)
        try:
            import pytesseract
            # Check if Tesseract binary is accessible
            pytesseract.get_tesseract_version()
        except ImportError:
            raise ImportError(
                "pytesseract library not installed. Run: pip install pytesseract"
            )
        except pytesseract.TesseractNotFoundError:
            raise RuntimeError(
                "Tesseract OCR binary not found. Install it:\n"
                "  Ubuntu/Debian: sudo apt-get install tesseract-ocr tesseract-ocr-ara\n"
                "  Windows: https://github.com/UB-Mannheim/tesseract/wiki\n"
                "  macOS: brew install tesseract tesseract-lang"
            )

    async def _extract_text_from_image(self, image: Image.Image) -> str:
        logger.info('Using TesseractProcessor')
        import pytesseract
        text = await asyncio.to_thread(
            pytesseract.image_to_string, image, lang='ara+eng'
        )
        return text