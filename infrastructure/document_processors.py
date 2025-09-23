# infrastructure/document_processors.py
"""Document processing implementations"""
import asyncio
import uuid
from typing import List
from PIL import Image
import io

from PIL import Image

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LangchainDocument

from core.interfaces import IDocumentProcessor, Chunk, IPdfToImageConverter
from config import settings

class BaseOCRProcessor(IDocumentProcessor):
    def __init__(self, pdf_converter: IPdfToImageConverter = None, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.pdf_converter = pdf_converter  # None for image-only processors
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
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
        
        # OCR all images
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
        if file_type.lower() == 'pdf':
            try:
                with open(file_path, 'rb') as f:
                    return f.read(4) == b'%PDF'
            except Exception:
                return False
        elif file_type.lower() in ['jpg', 'jpeg', 'png']:
            try:
                Image.open(file_path)
                return True
            except Exception:
                return False
        return False

class EasyOCRProcessor(BaseOCRProcessor):
    reader = None

    def __init__(self, pdf_converter: IPdfToImageConverter = None, **kwargs):  # Add = None
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
        text = "\n".join(result)
        print(f"--- EasyOCR Raw Output ---\n{text}\n--------------------------") # <-- ADD THIS LINE
        return text

class PaddleOCRProcessor(BaseOCRProcessor):
    ocr_engine = None

    def __init__(self, pdf_converter: IPdfToImageConverter = None, **kwargs):  # Add = None
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
        text = "\n".join(lines)
        print(f"--- PaddleOCR Raw Output ---\n{text}\n--------------------------") # <-- ADD THIS LINE
        return text