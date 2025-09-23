# infrastructure/document_processors.py
"""Document processing implementations"""
import asyncio
import uuid
from typing import List
from PIL import Image
import io

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LangchainDocument

from core.interfaces import IDocumentProcessor, Chunk, IPdfToImageConverter
from config import settings

class BaseOCRProcessor(IDocumentProcessor):
    def __init__(self, pdf_converter: IPdfToImageConverter = None, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.pdf_converter = pdf_converter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    async def process(self, file_path: str, file_type: str) -> List[Chunk]:
        # Get images based on file type
        if file_type.lower() == 'pdf':
            if not self.pdf_converter:
                raise ValueError("PDF converter required for PDF processing")
            images = await asyncio.to_thread(self.pdf_converter.convert, file_path, dpi=settings.OCR_DPI)
        elif file_type.lower() in ['jpg', 'jpeg', 'png']:
            images = [Image.open(file_path)]
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        # OCR all images and create documents
        docs = []
        for i, image in enumerate(images):
            text = await self._extract_text_from_image(image)
            if text.strip():  # Only add non-empty text
                docs.append(LangchainDocument(
                    page_content=text,
                    metadata={"page": i + 1, "source": file_path}
                ))

        if not docs:
            raise ValueError("No text extracted from document")

        # Split into chunks
        split_docs = self.text_splitter.split_documents(docs)
        return [
            Chunk(
                id=f"{uuid.uuid4()}_{i}",
                content=doc.page_content,
                document_id="",  # Set by service layer
                metadata={"page": doc.metadata.get("page", -1), "source": doc.metadata.get("source", "")}
            )
            for i, doc in enumerate(split_docs)
        ]
    
    async def validate(self, file_path: str, file_type: str) -> bool:
        try:
            if file_type.lower() == 'pdf':
                with open(file_path, 'rb') as f:
                    return f.read(4) == b'%PDF'
            elif file_type.lower() in ['jpg', 'jpeg', 'png']:
                Image.open(file_path).verify()
                return True
        except Exception:
            pass
        return False

    async def _extract_text_from_image(self, image: Image.Image) -> str:
        # Override in subclasses
        raise NotImplementedError

class EasyOCRProcessor(BaseOCRProcessor):
    reader = None

    def __init__(self, pdf_converter: IPdfToImageConverter = None, **kwargs):
        super().__init__(pdf_converter, **kwargs)
        if EasyOCRProcessor.reader is None:
            try:
                import easyocr
                EasyOCRProcessor.reader = easyocr.Reader(settings.OCR_LANGUAGES)
            except ImportError:
                raise ImportError("EasyOCR not installed. Run: pip install easyocr")

    async def _extract_text_from_image(self, image: Image.Image) -> str:
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='PNG')
        result = await asyncio.to_thread(
            self.reader.readtext, img_bytes.getvalue(), detail=0, paragraph=True
        )
        return "\n".join(result) if result else ""

class PaddleOCRProcessor(BaseOCRProcessor):
    ocr_engine = None

    def __init__(self, pdf_converter: IPdfToImageConverter = None, **kwargs):
        super().__init__(pdf_converter, **kwargs)
        if PaddleOCRProcessor.ocr_engine is None:
            try:
                from paddleocr import PaddleOCR
                lang = next((l for l in settings.OCR_LANGUAGES if l != 'en'), 'en')
                PaddleOCRProcessor.ocr_engine = PaddleOCR(use_angle_cls=True, lang=lang)
            except ImportError:
                raise ImportError("PaddleOCR not installed. Run: pip install paddleocr paddlepaddle")

    async def _extract_text_from_image(self, image: Image.Image) -> str:
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='PNG')
        result = await asyncio.to_thread(self.ocr_engine.ocr, img_bytes.getvalue(), cls=True)
        
        if not result or not result[0]:
            return ""
        
        return "\n".join(line[1][0] for line in result[0])