# infrastructure/document_processors.py
"""Document processing implementations (final, structure-first, Arabic-aware)

This module implements three OCR processors (Tesseract, EasyOCR, PaddleOCR) that all
feed a *structure-first* Arabic segmenter before length-based splitting.

Key properties:
- EasyOCR is forced into line-aware mode (detail=1, paragraph=False) and we rebuild
  lines via a compact y-clustering helper (see `infrastructure.line_builder`).
- PaddleOCR initialization is hardened with `_ensure_engine()` and raises a clear
  RuntimeError on failure so the caller can gracefully fallback to EasyOCR/Tesseract.
- The BaseOCRProcessor performs structure-aware presegmentation, preserves atomic
  boundaries for `header`/`item` segments, and only merges tiny *paragraph* continuations.
"""
from __future__ import annotations

import asyncio
import io
import logging
import uuid
from abc import abstractmethod
from typing import Any, Callable, List, Optional, Tuple

from PIL import Image

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LangchainDocument

from core.interfaces import IDocumentProcessor, DocumentChunk, IPdfToImageConverter
from api.schemas import DocumentProcessingError, ErrorCode
from config import settings

# External helpers
try:
    from utils.arabic_segmenter import segment_arabic  # type: ignore
except Exception:
    def segment_arabic(text: str):
        return [text]

try:
    from infrastructure.line_builder import rebuild_text_from_boxes  # type: ignore
except Exception:
    def rebuild_text_from_boxes(items, *, alpha_line: float = 0.7, beta_blank: float = 1.6) -> str:  # type: ignore
        return " ".join(t for _, t, _ in items) if items else ""

logger = logging.getLogger(settings.LOGGER_NAME)

# -----------------------------
# Configuration
# -----------------------------
ARABIC_AWARE_SEPARATORS: List[str] = [
    "\n\n", "\n", ":", "؛", "؟", ".", "!", "،", " ", ""
]

DEFAULT_CHUNK_SIZE = getattr(settings, "CHUNK_SIZE", 800)
DEFAULT_CHUNK_OVERLAP = getattr(settings, "CHUNK_OVERLAP", 60)
DEFAULT_MIN_CHUNK_CHARS = getattr(settings, "MIN_CHUNK_CHARS", 120)
PER_PAGE_TIMEOUT = getattr(settings, "OCR_TIMEOUT_SECONDS", 60)
OCR_DPI = getattr(settings, "OCR_DPI", 300)
IMAGE_EXTENSIONS = set(getattr(settings, "IMAGE_EXTENSIONS", [".png", ".jpg", ".jpeg"]))

# -----------------------------
# Arabic Text Utilities
# -----------------------------
def _has_arabic(s: str) -> bool:
    """Check if string contains Arabic characters."""
    return any('\u0600' <= ch <= '\u06FF' for ch in s)

def _has_latin(s: str) -> bool:
    """Check if string contains Latin letters."""
    return any(('A' <= ch <= 'Z') or ('a' <= ch <= 'z') for ch in s)

def _needs_flip_paddle_ar_token(s: str) -> bool:
    """
    PaddleOCR outputs Arabic tokens in LTR character order.
    Flip tokens that are Arabic-only (no Latin letters).
    """
    return _has_arabic(s) and not _has_latin(s)

def _flip_if_needed(s: str) -> str:
    """Reverse string if it's a PaddleOCR Arabic token."""
    return s[::-1] if _needs_flip_paddle_ar_token(s) else s

def _wrap_bidi_for_display(s: str) -> str:
    """
    Wrap Arabic-dominant lines with RTL embedding marks for proper display.
    U+202B (RLE) + text + U+202C (PDF) forces RTL rendering in LTR contexts.
    """
    lines = s.split("\n")
    out = []
    for ln in lines:
        if _has_arabic(ln) and not _has_latin(ln):
            out.append("\u202B" + ln + "\u202C")  # RTL embed
        else:
            out.append(ln)
    return "\n".join(out)

# -----------------------------
# Document Utilities
# -----------------------------
def _coerce_segment(part: Any) -> Tuple[str, str]:
    """Convert segment to (text, kind) tuple. Defaults kind to 'paragraph'."""
    if isinstance(part, tuple) and part:
        text = str(part[0]).strip()
        kind = str(part[1]).strip() if len(part) > 1 else "paragraph"
        return text, (kind or "paragraph")
    if isinstance(part, dict):
        text = str(part.get("text", "")).strip()
        kind = str(part.get("kind", "paragraph")).strip() or "paragraph"
        return text, kind
    return (str(part).strip(), "paragraph")

def _is_atomic(doc: LangchainDocument) -> bool:
    """Check if document segment is atomic (header/item) and shouldn't be merged."""
    kind = (doc.metadata or {}).get("segment_kind", "paragraph")
    return kind in {"item", "header"}

# -----------------------------
# Base Processor
# -----------------------------
class BaseOCRProcessor(IDocumentProcessor):
    """
    Base OCR processor that:
    - Converts PDFs to images (if needed)
    - Extracts text per page using engine-specific _extract_text_from_image
    - Segments Arabic text by structure (list items, headers) before splitting by length
    - Preserves 'atomic' boundaries (header/item) during tiny-merge
    """
    def __init__(
        self,
        pdf_converter: Optional[IPdfToImageConverter] = None,
        *,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> None:
        self.pdf_converter = pdf_converter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=ARABIC_AWARE_SEPARATORS
        )
        self.per_page_timeout_s = PER_PAGE_TIMEOUT

    @abstractmethod
    async def _extract_text_from_image(self, image: Image.Image) -> str:
        """Engine-specific page OCR -> plain text with reliable newlines."""
        raise NotImplementedError

    async def process(
        self,
        file_path: str,
        file_type: str,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[DocumentChunk]:
        # 1) Build list of page images
        images = await self._load_images(file_path, file_type)
        total_pages = len(images)
        
        # 2) Extract text from each page
        docs: List[LangchainDocument] = []
        for i, image in enumerate(images):
            self._update_progress(progress_callback, i + 1, total_pages)
            
            text = await self._extract_page_text(image, i + 1)
            if not text:
                continue
            
            # Structure-first segmentation
            segments = self._segment_text(text, file_path, i + 1)
            docs.extend(segments)

        if not docs:
            raise DocumentProcessingError(
                "No text extracted from document",
                ErrorCode.NO_TEXT_FOUND
            )

        # 3) Split, merge, and wrap
        split_docs = self.text_splitter.split_documents(docs)
        merged_docs = self._merge_tiny_paragraphs(split_docs)
        
        logger.info("Processed %d raw segments into %d chunks", len(docs), len(merged_docs))

        # 4) Build final chunks with optional display wrapping
        chunks = self._build_chunks(merged_docs)
        return chunks

    async def _load_images(self, file_path: str, file_type: str) -> List[Image.Image]:
        """Load images from PDF or image file."""
        if file_type == "pdf":
            if not self.pdf_converter:
                raise DocumentProcessingError(
                    "PDF converter required for PDF processing",
                    ErrorCode.PROCESSING_FAILED
                )
            try:
                return await asyncio.to_thread(
                    self.pdf_converter.convert, file_path, dpi=OCR_DPI
                )
            except Exception as e:
                logger.exception("Failed converting PDF to images")
                raise DocumentProcessingError(
                    f"Failed to convert PDF to images: {e}",
                    ErrorCode.PROCESSING_FAILED
                )
        
        elif file_type.lower() in IMAGE_EXTENSIONS or any(
            file_path.lower().endswith(ext) for ext in IMAGE_EXTENSIONS
        ):
            try:
                with Image.open(file_path) as img:
                    return [img.convert("RGB")]
            except Exception as e:
                logger.exception("Failed opening image")
                raise DocumentProcessingError(
                    f"Failed to open image: {e}",
                    ErrorCode.PROCESSING_FAILED
                )
        
        raise DocumentProcessingError(
            f"Unsupported file type: {file_type}",
            ErrorCode.INVALID_FORMAT
        )

    def _update_progress(
        self, 
        callback: Optional[Callable[[int, int], None]], 
        current: int, 
        total: int
    ) -> None:
        """Call progress callback if provided."""
        if callback:
            try:
                callback(current, total)
            except Exception:
                pass  # Don't break OCR if UI progress fails

    async def _extract_page_text(self, image: Image.Image, page_num: int) -> str:
        """Extract text from single page with timeout and error handling."""
        try:
            return await asyncio.wait_for(
                self._extract_text_from_image(image),
                timeout=self.per_page_timeout_s
            )
        except asyncio.TimeoutError:
            logger.warning("OCR timed out on page %d", page_num)
            raise DocumentProcessingError(
                f"OCR timeout on page {page_num}",
                ErrorCode.OCR_TIMEOUT
            )
        except RuntimeError:
            raise  # Allow caller to catch engine-specific failures
        except Exception as e:
            logger.exception("Unexpected OCR exception on page %d", page_num)
            raise DocumentProcessingError(
                f"OCR failed on page {page_num}: {e}",
                ErrorCode.PROCESSING_FAILED
            )

    def _segment_text(
        self, 
        text: str, 
        source: str, 
        page: int
    ) -> List[LangchainDocument]:
        """Segment text into structured parts (headers, items, paragraphs)."""
        try:
            parts = segment_arabic(text)
        except Exception:
            parts = [text]  # Fallback to whole page

        docs = []
        for part in parts:
            seg_text, seg_kind = _coerce_segment(part)
            if not seg_text:
                continue
            docs.append(
                LangchainDocument(
                    page_content=seg_text,
                    metadata={
                        "page": page,
                        "source": source,
                        "segment_kind": seg_kind
                    }
                )
            )
        return docs

    def _merge_tiny_paragraphs(
        self, 
        docs: List[LangchainDocument]
    ) -> List[LangchainDocument]:
        """Merge tiny paragraph continuations (protect atomic boundaries)."""
        merged: List[LangchainDocument] = []
        
        for doc in docs:
            should_merge = (
                merged
                and len(merged[-1].page_content) < DEFAULT_MIN_CHUNK_CHARS
                and not _is_atomic(merged[-1])
                and not _is_atomic(doc)
                and (merged[-1].metadata or {}).get("segment_kind") == "paragraph"
                and (doc.metadata or {}).get("segment_kind") == "paragraph"
            )
            
            if should_merge:
                merged[-1].page_content = (
                    merged[-1].page_content.rstrip() + "\n" + 
                    doc.page_content.lstrip()
                ).strip()
            else:
                merged.append(doc)
        
        return merged

    def _build_chunks(self, docs: List[LangchainDocument]) -> List[DocumentChunk]:
        """Build final DocumentChunk objects (clean content, no display formatting)."""
        chunks: List[DocumentChunk] = []
        for i, doc in enumerate(docs):
            chunks.append(
                DocumentChunk(
                    id=f"{uuid.uuid4()}_{i}",
                    content=doc.page_content,  # Clean content
                    document_id="",
                    metadata={
                        "page": (doc.metadata or {}).get("page", -1),
                        "source": (doc.metadata or {}).get("source", ""),
                        "segment_kind": (doc.metadata or {}).get("segment_kind", "paragraph")
                    }
                )
            )
        return chunks

# -----------------------------
# Tesseract
# -----------------------------
class TesseractProcessor(BaseOCRProcessor):
    """Tesseract OCR engine (ara+eng)."""
    
    async def _extract_text_from_image(self, image: Image.Image) -> str:
        logger.info("Using TesseractProcessor")
        try:
            import pytesseract
        except Exception as e:
            raise RuntimeError(
                "Tesseract not available. Install tesseract-ocr with Arabic data."
            ) from e

        text = await asyncio.to_thread(
            pytesseract.image_to_string, image, lang="ara+eng"
        )
        return text

# -----------------------------
# EasyOCR
# -----------------------------
class EasyOCRProcessor(BaseOCRProcessor):
    """EasyOCR with line-aware output (detail=1, paragraph=False)."""
    
    reader: Optional[Any] = None

    async def _init_reader(self) -> None:
        if EasyOCRProcessor.reader is None:
            try:
                import easyocr  # type: ignore
            except Exception as e:
                raise RuntimeError(
                    "EasyOCR not installed. Run: pip install easyocr"
                ) from e
            
            EasyOCRProcessor.reader = easyocr.Reader(
                getattr(settings, "OCR_LANGUAGES", ["ar", "en"])
            )

    async def _extract_text_from_image(self, image: Image.Image) -> str:
        logger.info("Using EasyOCRProcessor")
        await self._init_reader()
        
        img_bytes = io.BytesIO()
        image.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        result = await asyncio.to_thread(
            EasyOCRProcessor.reader.readtext,  # type: ignore[attr-defined]
            img_bytes.getvalue(),
            detail=1,
            paragraph=False
        )

        # Extract items: (bbox, text, conf)
        items: List[Tuple[List[List[int]], str, float]] = []
        for entry in result or []:
            if not entry or len(entry) < 2:
                continue
            bbox = entry[0]
            text = entry[1] if len(entry) > 1 else ""
            conf = float(entry[2]) if len(entry) > 2 else 0.0
            if text:
                items.append((bbox, text, conf))

        return rebuild_text_from_boxes(items)

# -----------------------------
# PaddleOCR
# -----------------------------
class PaddleOCRProcessor(BaseOCRProcessor):
    """
    PaddleOCR with Arabic LTR token normalization.
    Handles both dict and list result formats.
    """
    
    def __init__(self, pdf_converter: Optional[IPdfToImageConverter] = None, **kwargs) -> None:
        super().__init__(pdf_converter, **kwargs)
        self._engine = None

    def _ensure_engine(self) -> None:
        """Initialize PaddleOCR with version/platform checks."""
        import sys

        if self._engine is not None:
            return
        
        try:
            if sys.platform.startswith("win") and sys.version_info >= (3, 13):
                raise RuntimeError(
                    "PaddleOCR is not supported on Python 3.13 on Windows. "
                    "Use Python 3.10 or 3.11 for PaddleOCR on Windows."
                )
            
            from paddleocr import PaddleOCR  # type: ignore
            self._engine = PaddleOCR(
                use_angle_cls=True,
                lang=getattr(settings, "PADDLE_OCR_LANG", "ar")
            )
        except Exception as e:
            logger.exception("PaddleOCR initialization failed")
            self._engine = None
            raise RuntimeError(
                "PaddleOCR initialization failed. Ensure a supported Python version "
                "(3.10/3.11 on Windows) and install compatible paddlepaddle/paddleocr."
            ) from e

    async def _extract_text_from_image(self, image: Image.Image) -> str:
        logger.info("Using PaddleOCRProcessor")
        self._ensure_engine()

        # Convert to NumPy array
        import numpy as np
        img_array = np.array(image.convert("RGB"))

        # Run OCR
        result = await asyncio.to_thread(self._engine.ocr, img_array)  # type: ignore

        # Extract items from result
        items = self._parse_paddle_result(result)
        
        # Fix Arabic LTR tokens (PaddleOCR quirk)
        items = [(bbox, _flip_if_needed(text), conf) for (bbox, text, conf) in items]
        
        # Rebuild text with proper line ordering
        return rebuild_text_from_boxes(items)

    def _parse_paddle_result(
        self, 
        result: Any
    ) -> List[Tuple[List[List[int]], str, float]]:
        """
        Parse PaddleOCR result (handles both dict and list formats).
        Returns: [(bbox, text, conf), ...]
        """
        items: List[Tuple[List[List[int]], str, float]] = []

        # Format 1: Direct dict
        if isinstance(result, dict):
            self._extract_from_dict(result, items)
        
        # Format 2: List (may contain dict or nested lists)
        elif isinstance(result, list) and result:
            first_page = result[0]
            
            if isinstance(first_page, dict):
                # List of dicts (one per page)
                self._extract_from_dict(first_page, items)
            elif isinstance(first_page, list):
                # Classic nested list format
                self._extract_from_nested_list(result, items)
        
        logger.info(f"Extracted {len(items)} text items from PaddleOCR")
        return items

    def _extract_from_dict(
        self, 
        data: dict, 
        items: List[Tuple[List[List[int]], str, float]]
    ) -> None:
        """Extract items from dict-based PaddleOCR result."""
        texts = data.get('rec_texts') or data.get('texts') or []
        scores = data.get('rec_scores') or data.get('scores') or []
        polys = data.get('rec_polys') or data.get('polys') or []
        
        for i, text in enumerate(texts):
            if not text or not text.strip():
                continue
            
            score = float(scores[i]) if i < len(scores) else 0.0
            
            if i < len(polys):
                bbox_raw = polys[i]
                bbox = bbox_raw.tolist() if hasattr(bbox_raw, 'tolist') else bbox_raw
            else:
                bbox = []
            
            if bbox and text.strip():
                items.append((bbox, text.strip(), score))

    def _extract_from_nested_list(
        self, 
        result: list, 
        items: List[Tuple[List[List[int]], str, float]]
    ) -> None:
        """Extract items from classic nested list format."""
        for page in result:
            if not page:
                continue
            for detection in page:
                if not detection or len(detection) < 2:
                    continue
                
                bbox = detection[0]
                text_conf = detection[1]
                
                if isinstance(text_conf, (list, tuple)) and len(text_conf) >= 1:
                    text = text_conf[0] if isinstance(text_conf[0], str) else str(text_conf[0])
                    conf = float(text_conf[1]) if len(text_conf) > 1 else 0.0
                    
                    if text and text.strip():
                        items.append((bbox, text, conf))