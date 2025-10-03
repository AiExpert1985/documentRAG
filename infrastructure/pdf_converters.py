# infrastructure/pdf_converters.py
"""Concrete implementations for PDF to image conversion."""
from typing import List
import fitz  # PyMuPDF
from PIL import Image
import io

from core.interfaces import IPdfToImageConverter

class PyMuPDFConverter(IPdfToImageConverter):
    def convert(self, file_path: str, dpi: int = 300) -> List[Image.Image]:
        """Converts PDF pages to PIL Images."""
        images = []
        doc = fitz.open(file_path)
        for page in doc:
            pix = page.get_pixmap(dpi=dpi) # type: ignore
            img = Image.open(io.BytesIO(pix.tobytes()))
            images.append(img)
        doc.close()
        return images