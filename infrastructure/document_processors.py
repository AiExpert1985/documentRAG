# infrastructure/document_processors.py
"""Document processing implementations"""
from abc import abstractmethod
import asyncio
import uuid
from typing import Any, List, Optional
from PIL import Image
import io

import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LangchainDocument

from core.interfaces import IDocumentProcessor, DocumentChunk, IPdfToImageConverter
from config import settings


logger = logging.getLogger(settings.LOGGER_NAME)

class BaseOCRProcessor(IDocumentProcessor):
    def __init__(self, pdf_converter: Optional[IPdfToImageConverter] = None, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.pdf_converter = pdf_converter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n",   # Separate paragraphs first (strongest separator)
                "\n",     # Separate lines 
                " ",      # Separate spaces (if needed, but usually last)
                ".",      # Separate sentences
                "؟",      # Arabic question mark (new)
                "!",      # Exclamation mark (new)
                "،",      # Arabic comma
                ""        # Fallback to character split
            ] 
        )

    @abstractmethod
    async def _extract_text_from_image(self, image: Image.Image) -> str:
        """
        Extracts text from a single image using the configured OCR engine.
        
        Args:
            image: PIL (Python Imaging Library) Image object containing the visual content to process
                (either a direct image file or a page converted from PDF)
            
        Returns:
            str: Extracted text content from the image. Empty string if no text found.
            
        Raises:
            NotImplementedError: If called on base class without override
            asyncio.TimeoutError: If OCR processing exceeds configured timeout
            ImportError: If required OCR library is not installed
            
        Implementation Notes:
            - Must handle both single-language and multi-language text
            - Should return empty string rather than None when no text detected
            - Should preserve text structure (paragraphs, line breaks) when possible
            - Runs in async context to avoid blocking during OCR processing
            
        Example Implementations:
            EasyOCRProcessor: Uses easyocr.Reader.readtext()
            PaddleOCRProcessor: Uses PaddleOCR.ocr()
            
        Note:
            This method is called inside a timeout wrapper (OCR_TIMEOUT_SECONDS)
            to prevent hanging on corrupted or extremely large images. OCR engines
            are initialized once as class variables and reused across all instances
            for performance.
        """
        pass

    async def process(self, file_path: str, file_type: str) -> List[DocumentChunk]:
        """
        Extracts text from a document file and splits it into searchable chunks.
        
        Handles both PDF files (by converting to images first) and direct image files.
        Uses OCR to extract text from each page/image, then splits the extracted text
        into smaller chunks suitable for semantic search and embedding generation.
        
        Args:
            file_path: Absolute path to the document file on disk
            file_type: Normalized file extension ("pdf", "jpg", "jpeg", "png")
                    Must be lowercase without dot
            
        Returns:
            List[DocumentChunk]: Text chunks ready for embedding, each containing:
                - Extracted text content
                - Page number metadata
                - Unique chunk ID
                - Empty document_id (set by service layer)
                
        Raises:
            ValueError: If file type is unsupported, PDF converter missing for PDFs,
                    OCR timeout exceeded, or no text extracted from document
            
        Process Flow:
            1. Load images (convert PDF pages or open image file)
            2. Extract text from each image using OCR (with timeout protection)
            3. Split extracted text into chunks using RecursiveCharacterTextSplitter
            4. Return chunks with metadata (page numbers, source path)
            
        Note:
            OCR extraction has a configurable timeout (OCR_TIMEOUT_SECONDS) per page
            to prevent hanging on corrupted or oversized files. Chunk size and overlap
            are configured via CHUNK_SIZE and CHUNK_OVERLAP settings.
        """
        
        if file_type == 'pdf':
            if not self.pdf_converter:
                raise ValueError("PDF converter required for PDF processing")
            images = await asyncio.to_thread(self.pdf_converter.convert, file_path, dpi=settings.OCR_DPI)
        elif file_type in settings.IMAGE_EXTENSIONS: # Use config for comparison
            images = [Image.open(file_path)]
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        docs : List[LangchainDocument] = []
        for i, image in enumerate(images):
            try:
                text = await asyncio.wait_for(
                    self._extract_text_from_image(image),
                    timeout=settings.OCR_TIMEOUT_SECONDS
                )
            except asyncio.TimeoutError:
                raise ValueError(f"OCR timeout on page {i+1} after {settings.OCR_TIMEOUT_SECONDS}s")
            
            # This ensures the splitter treats bullet points as separate documents.
            # Split the page text by strong separators (paragraph breaks)
            if text.strip(): 
                # Split the page text by strong separators (paragraph breaks)
                # This logic relies on EasyOCR's output having \n\n breaks
                page_sections = text.split('\n\n')
                
                for section in page_sections:
                    cleaned_section = section.strip()
                    
                    MIN_CHUNK_LENGTH = 30 # Define a reasonable minimum length
                    
                    if len(cleaned_section) >= MIN_CHUNK_LENGTH:
                        docs.append(LangchainDocument(
                            page_content=cleaned_section,
                            metadata={"page": i + 1, "source": file_path} 
                        ))


        if not docs:
            raise ValueError("No text extracted from document")
        
        print("Before split")
        for i, doc in enumerate(docs):
            print(f"{i} - Length: {len(doc.page_content)} chars")

        split_docs = self.text_splitter.split_documents(docs)

        print(f"Text length: {len(docs[0].page_content)}")
        print(f"Chunk size setting: {self.text_splitter._chunk_size}")
        print(f"Chunks created: {len(split_docs)}")

        print("After split")
        for i, doc in enumerate(split_docs):
            print(f"{i} - Length: {len(doc.page_content)} chars")

        logger.info(f"Successfully processed document")
        return [
            DocumentChunk(
                id=f"{uuid.uuid4()}_{i}",
                content=doc.page_content,
                document_id="",  # Set by service layer
                metadata={"page": doc.metadata.get("page", -1), "source": doc.metadata.get("source", "")}
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
        result = await asyncio.to_thread(
            self.reader.readtext, img_bytes.getvalue(), detail=0, paragraph=True
        )
        return "\n".join(result) if result else ""

class PaddleOCRProcessor(BaseOCRProcessor):
    ocr_engine: Optional[Any] = None
            
    async def _extract_text_from_image(self, image: Image.Image) -> str:
        assert self.ocr_engine is not None, "PaddleOCR engine not initialized"
        logger.info('Using PaddleOCRProcessor')
        import numpy as np
        
        img_array = np.array(image)
        
        result = await asyncio.to_thread(
            self.ocr_engine.ocr, img_array
        )
        
        print(f"PaddleOCR raw result type: {type(result)}")
        print(f"PaddleOCR raw result: {result}")
        
        if not result:
            return ""
        
        # PaddleOCR returns different structure - adapt based on actual output
        try:
            # Try to extract text - structure might be different
            if isinstance(result, list) and len(result) > 0:
                if result[0] is None:
                    return ""
                text = "\n".join(line[1][0] for line in result[0])
            else:
                return ""
        except (IndexError, TypeError) as e:
            print(f"Failed to parse PaddleOCR result: {e}")
            print(f"Result structure: {result}")
            return ""
        
        print(f"Extracted text length: {len(text)}")
        return text
    

class TesseractProcessor(BaseOCRProcessor):
    def __init__(self, pdf_converter: Optional[IPdfToImageConverter] = None, **kwargs):
        super().__init__(pdf_converter, **kwargs)
        try:
            import pytesseract
            # Set path if needed (Windows)
            # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        except ImportError:
            raise ImportError("Tesseract not installed")

    async def _extract_text_from_image(self, image: Image.Image) -> str:
        logger.info('Using TesseractProcessor')
        import pytesseract
        text = await asyncio.to_thread(
            pytesseract.image_to_string, image, lang='ara+eng'
        )
        return text