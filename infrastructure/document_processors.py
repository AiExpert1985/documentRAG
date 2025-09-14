# infrastructure/document_processors.py
"""Document processing implementations"""
import asyncio
import uuid
from typing import List
from PIL import Image
import io

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_core.documents import Document as LangchainDocument

from core.interfaces import IDocumentProcessor, Chunk, IPdfToImageConverter
from .pdf_converters import PyMuPDFConverter
from config import settings

class BaseOCRProcessor(IDocumentProcessor):
    """Abstract base class for OCR processors that work on images."""
    
    def __init__(
        self, 
        pdf_converter: IPdfToImageConverter,
        chunk_size: int = 1000, 
        chunk_overlap: int = 200
    ):
        self.pdf_converter = pdf_converter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    async def _extract_text_from_image(self, image: Image.Image) -> str:
        raise NotImplementedError("Subclasses must implement this method.")

    async def process(self, file_path: str, file_type: str) -> List[Chunk]:
        if file_type.lower() != 'pdf':
            raise ValueError(f"This processor only handles PDF files for OCR, not {file_type}")

        images = await asyncio.to_thread(self.pdf_converter.convert, file_path, dpi=settings.OCR_DPI)
        
        full_text_per_page = []
        for i, image in enumerate(images):
            text = await self._extract_text_from_image(image)
            doc = LangchainDocument(
                page_content=text,
                metadata={"page": i + 1, "source": file_path}
            )
            full_text_per_page.append(doc)

        split_docs = self.text_splitter.split_documents(full_text_per_page)
        
        chunks = []
        for i, doc in enumerate(split_docs):
            chunk_id = f"{uuid.uuid4()}_{i}"
            chunks.append(Chunk(
                id=chunk_id,
                content=doc.page_content,
                document_id="",
                metadata={
                    "page": doc.metadata.get("page", -1),
                    "source": doc.metadata.get("source", "")
                }
            ))
        return chunks
    
    async def validate(self, file_path: str, file_type: str) -> bool:
        if file_type.lower() != 'pdf':
            return False
        try:
            with open(file_path, 'rb') as f:
                return f.read(4) == b'%PDF'
        except Exception:
            return False

class EasyOCRProcessor(BaseOCRProcessor):
    reader = None

    def __init__(self, pdf_converter: IPdfToImageConverter, **kwargs):
        super().__init__(pdf_converter, **kwargs)
        try:
            if EasyOCRProcessor.reader is None:
                import easyocr
                EasyOCRProcessor.reader = easyocr.Reader(settings.OCR_LANGUAGES)
        except ImportError:
            raise ImportError("EasyOCR not installed. Run: pip install easyocr")

    async def _extract_text_from_image(self, image: Image.Image) -> str:
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        result = await asyncio.to_thread(self.reader.readtext, img_byte_arr, detail=0, paragraph=True)
        return "\n".join(result)

class PaddleOCRProcessor(BaseOCRProcessor):
    ocr_engine = None

    def __init__(self, pdf_converter: IPdfToImageConverter, **kwargs):
        super().__init__(pdf_converter, **kwargs)
        try:
            if PaddleOCRProcessor.ocr_engine is None:
                from paddleocr import PaddleOCR
                # Note: PaddleOCR uses 'ch' for Chinese, 'en' for English, 'ar' for Arabic, etc.
                # It doesn't support a list of languages in the same way as EasyOCR.
                # We select the first language from the list that isn't English, or default to English.
                lang = next((lang for lang in settings.OCR_LANGUAGES if lang != 'en'), 'en')
                PaddleOCRProcessor.ocr_engine = PaddleOCR(use_angle_cls=True, lang=lang)
        except ImportError:
            raise ImportError("PaddleOCR not installed. Run: pip install paddleocr paddlepaddle")

    async def _extract_text_from_image(self, image: Image.Image) -> str:
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        result = await asyncio.to_thread(self.ocr_engine.ocr, img_byte_arr, cls=True)
        lines = [line[1][0] for res_part in result for line in res_part]
        return "\n".join(lines)

class PDFProcessor(IDocumentProcessor):
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    async def process(self, file_path: str, file_type: str) -> List[Chunk]:
        if file_type.lower() != 'pdf':
            raise ValueError(f"PDFProcessor cannot handle {file_type} files")
        
        loader = UnstructuredPDFLoader(file_path, mode="elements", strategy="hi_res")
        documents = await asyncio.to_thread(loader.load_and_split, self.text_splitter)
        
        if not documents:
            raise ValueError("No content extracted from PDF")
        
        chunks = []
        for i, doc in enumerate(documents):
            chunk_id = f"{uuid.uuid4()}_{i}"
            chunks.append(Chunk(
                id=chunk_id,
                content=doc.page_content,
                document_id="",
                metadata={
                    "page": doc.metadata.get("page_number", -1),
                    "source": doc.metadata.get("source", "")
                }
            ))
        return chunks
    
    async def validate(self, file_path: str, file_type: str) -> bool:
        if file_type.lower() != 'pdf':
            return False
        try:
            with open(file_path, 'rb') as f:
                return f.read(4) == b'%PDF'
        except Exception:
            return False