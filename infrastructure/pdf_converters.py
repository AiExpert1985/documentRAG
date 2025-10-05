# infrastructure/pdf_converters.py
"""Concrete implementations for PDF to image conversion."""
from typing import List
import fitz  # PyMuPDF
from PIL import Image
import io

from core.interfaces import IPdfToImageConverter

class PyMuPDFConverter(IPdfToImageConverter):  # ← This class exists?
    def convert(self, file_path: str, dpi: int = 300) -> List[Image.Image]:
        """Converts PDF pages to PIL Images."""
        images = []
        with fitz.open(file_path) as doc:
            for page in doc:
                pix = page.get_pixmap(dpi=dpi)  # type: ignore
                img = Image.open(io.BytesIO(pix.tobytes())).copy()
                images.append(img)
        return images