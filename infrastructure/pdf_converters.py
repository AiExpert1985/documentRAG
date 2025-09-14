# infrastructure/pdf_converters.py
"""Concrete implementations for PDF to image conversion."""
from typing import List
import fitz  # PyMuPDF
from PIL import Image
import io

from core.interfaces import IPdfToImageConverter

class PyMuPDFConverter(IPdfToImageConverter):
    """Converts PDF to a list of PIL Images using PyMuPDF."""

    def convert(self, file_path: str, dpi: int = 300) -> List[Image.Image]:
        """
        Converts each page of a PDF into a PIL Image.

        Args:
            file_path (str): The path to the PDF file.
            dpi (int): The resolution to render the images at.

        Returns:
            List[Image.Image]: A list of PIL Image objects.
        """
        images = []
        doc = fitz.open(file_path)
        for page in doc:
            pix = page.get_pixmap(dpi=dpi)
            img = Image.open(io.BytesIO(pix.tobytes()))
            images.append(img)
        doc.close()
        return images