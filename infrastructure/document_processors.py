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
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from PIL import Image

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LangchainDocument

from core.interfaces import IDocumentProcessor, DocumentChunk, IPdfToImageConverter
from api.schemas import DocumentProcessingError, ErrorCode
from config import settings

# External helpers we introduced:
# - utils.arabic_segmenter.segment_arabic: returns either List[str] or List[Tuple[str, kind]] or List[{"text":..., "kind":...}]
# - infrastructure.line_builder.rebuild_text_from_boxes: rebuild lines from OCR boxes
try:
    from utils.arabic_segmenter import segment_arabic  # type: ignore
except Exception:
    # Fallback: pass-through segmentation
    def segment_arabic(text: str):
        return [text]

try:
    from infrastructure.line_builder import rebuild_text_from_boxes  # type: ignore
except Exception:
    def rebuild_text_from_boxes(items, *, alpha_line: float = 0.7, beta_blank: float = 1.6) -> str:  # type: ignore
        # Extremely small fallback: join tokens sorted by their insertion order.
        return " ".join(t for _, t, _ in items) if items else ""


logger = logging.getLogger(settings.LOGGER_NAME)

# -----------------------------
# Utilities
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

def _coerce_segment(part: Any) -> Tuple[str, str]:
    """Accept segment as str | (text, kind) | dict and return (text, kind).
    Defaults kind to 'paragraph' when unknown.
    """
    if isinstance(part, tuple) and part:
        text = str(part[0]).strip()
        kind = str(part[1]).strip() if len(part) > 1 else "paragraph"
        return text, (kind or "paragraph")
    if isinstance(part, dict):
        text = str(part.get("text", "")).strip()
        kind = str(part.get("kind", "paragraph")).strip() or "paragraph"
        return text, kind
    # Fallback assume raw text
    return (str(part).strip(), "paragraph")


def _is_atomic(doc: LangchainDocument) -> bool:
    kind = (doc.metadata or {}).get("segment_kind", "paragraph")
    return kind in {"item", "header"}


def _has_arabic(s: str) -> bool:
    return any('\u0600' <= ch <= '\u06FF' for ch in s)

def _has_latin(s: str) -> bool:
    return any(('A' <= ch <= 'Z') or ('a' <= ch <= 'z') for ch in s)

def _needs_flip_paddle_ar_token(s: str) -> bool:
    return _has_arabic(s) and not _has_latin(s)

def _flip_if_needed(s: str) -> str:
    return s[::-1] if _needs_flip_paddle_ar_token(s) else s

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
        if file_type == "pdf":
            if not self.pdf_converter:
                raise DocumentProcessingError(
                    "PDF converter required for PDF processing",
                    ErrorCode.PROCESSING_FAILED
                )
            try:
                images = await asyncio.to_thread(
                    self.pdf_converter.convert, file_path, dpi=OCR_DPI
                )
            except Exception as e:
                logger.exception("Failed converting PDF to images")
                raise DocumentProcessingError(
                    f"Failed to convert PDF to images: {e}",
                    ErrorCode.PROCESSING_FAILED
                )
        elif file_type.lower() in IMAGE_EXTENSIONS or any(file_path.lower().endswith(ext) for ext in IMAGE_EXTENSIONS):
            try:
                with Image.open(file_path) as img:
                    images = [img.convert("RGB")]
            except Exception as e:
                logger.exception("Failed opening image")
                raise DocumentProcessingError(
                    f"Failed to open image: {e}",
                    ErrorCode.PROCESSING_FAILED
                )
        else:
            raise DocumentProcessingError(
                f"Unsupported file type: {file_type}",
                ErrorCode.INVALID_FORMAT
            )

        total_pages = len(images)
        docs: List[LangchainDocument] = []

        # 2) Per page OCR + structure-first segmentation
        for i, image in enumerate(images):
            # progress update
            if progress_callback:
                try:
                    progress_callback(i + 1, total_pages)
                except Exception:
                    # Don't break OCR if UI progress fails
                    pass

            try:
                text = await asyncio.wait_for(
                    self._extract_text_from_image(image),
                    timeout=self.per_page_timeout_s
                )
            except asyncio.TimeoutError:
                logger.warning("OCR timed out on page %d", i + 1)
                raise DocumentProcessingError(
                    f"OCR timeout on page {i+1}",
                    ErrorCode.OCR_TIMEOUT
                )
            except RuntimeError as e:
                # Allow caller/factory to catch engine-specific failures and fallback
                raise
            except Exception as e:
                logger.exception("Unexpected OCR exception on page %d", i + 1)
                raise DocumentProcessingError(
                    f"OCR failed on page {i+1}: {e}",
                    ErrorCode.PROCESSING_FAILED
                )

            if not text or not text.strip():
                continue

            # Structure-first segmentation (returns list of segments)
            try:
                parts = segment_arabic(text)
            except Exception:
                # If segmenter fails, fallback to whole page
                parts = [text]

            for part in parts:
                seg_text, seg_kind = _coerce_segment(part)
                if not seg_text:
                    continue
                docs.append(
                    LangchainDocument(
                        page_content=seg_text,
                        metadata={
                            "page": i + 1,
                            "source": file_path,
                            "segment_kind": seg_kind
                        }
                    )
                )

        if not docs:
            raise DocumentProcessingError(
                "No text extracted from document",
                ErrorCode.NO_TEXT_FOUND
            )

        # 3) Length split (only for long segments)
        split_docs = self.text_splitter.split_documents(docs)

        # 4) Tiny merge for paragraph-only continuations (protect atomic boundaries)
        merged: List[LangchainDocument] = []
        for d in split_docs:
            if (
                merged
                and len(merged[-1].page_content) < DEFAULT_MIN_CHUNK_CHARS
                and not _is_atomic(merged[-1])
                and not _is_atomic(d)
                and (merged[-1].metadata or {}).get("segment_kind", "paragraph") == "paragraph"
                and (d.metadata or {}).get("segment_kind", "paragraph") == "paragraph"
            ):
                merged[-1].page_content = (merged[-1].page_content.rstrip() + "\n" + d.page_content.lstrip()).strip()
            else:
                merged.append(d)

        split_docs = merged

        logger.info("Processed %d raw segments into %d chunks", len(docs), len(split_docs))

        # 5) Build DocumentChunk payloads (document_id filled later in service)
        chunks: List[DocumentChunk] = [
            DocumentChunk(
                id=f"{uuid.uuid4()}_{i}",
                content=doc.page_content,
                document_id="",
                metadata={
                    "page": (doc.metadata or {}).get("page", -1),
                    "source": (doc.metadata or {}).get("source", ""),
                    "segment_kind": (doc.metadata or {}).get("segment_kind", "paragraph")
                }
            )
            for i, doc in enumerate(split_docs)
        ]
        return chunks


# -----------------------------
# Tesseract
# -----------------------------
class TesseractProcessor(BaseOCRProcessor):
    """Tesseract OCR engine (ara+eng). Keeps original per-line layout reasonably well."""
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
        # pytesseract returns text with \n lines; leave as-is for segmenter
        return text


# -----------------------------
# EasyOCR
# -----------------------------
class EasyOCRProcessor(BaseOCRProcessor):
    """
    EasyOCR configured for line-aware output. We request detail=1 & paragraph=False,
    then rebuild stable lines from bounding boxes using the shared line_builder helper.
    """
    reader: Optional[Any] = None

    async def _init_reader(self) -> None:
        if EasyOCRProcessor.reader is None:
            try:
                import easyocr  # type: ignore
            except Exception as e:
                raise RuntimeError(
                    "EasyOCR not installed. Run: pip install easyocr"
                ) from e
            # paragraph=False keeps tokens separate; we'll cluster into lines
            EasyOCRProcessor.reader = easyocr.Reader(
                getattr(settings, "OCR_LANGUAGES", ["ar", "en"]),
            )

    async def _extract_text_from_image(self, image: Image.Image) -> str:
        logger.info("Using EasyOCRProcessor")
        await self._init_reader()
        img_bytes = io.BytesIO()
        image.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        # detail=1 -> [(bbox, text, conf), ...]; paragraph=False -> avoid pre-merged blocks
        result = await asyncio.to_thread(
            EasyOCRProcessor.reader.readtext,  # type: ignore[attr-defined]
            img_bytes.getvalue(),
            detail=1,
            paragraph=False
        )

        # Normalize to list of (bbox, text, conf)
        items: List[Tuple[List[List[int]], str, float]] = []
        for entry in result or []:
            if not entry or len(entry) < 2:
                continue
            bbox = entry[0]
            text = entry[1] if len(entry) > 1 else ""
            conf = float(entry[2]) if len(entry) > 2 else 0.0
            if text:
                items.append((bbox, text, conf))

        # Rebuild lines with preserved blank lines
        text = rebuild_text_from_boxes(items)
        return text


# -----------------------------
# PaddleOCR
# -----------------------------
class PaddleOCRProcessor(BaseOCRProcessor):
    """
    PaddleOCR with hardened initialization and the same line rebuild as EasyOCR.
    If initialization fails (e.g., unsupported Python on Windows), _ensure_engine()
    raises RuntimeError so the caller can fallback to other engines.
    """
    def __init__(self, pdf_converter: Optional[IPdfToImageConverter] = None, **kwargs) -> None:
        super().__init__(pdf_converter, **kwargs)
        self._engine = None
        

    def _ensure_engine(self) -> None:
        # inside PaddleOCRProcessor._ensure_engine()
        import sys

        if self._engine is not None:
            return
        try:
            # Guard: current Windows wheels for PaddleOCR often don't support Python 3.13
            if sys.platform.startswith("win") and sys.version_info >= (3, 13):
                raise RuntimeError(
                    "PaddleOCR is not supported on Python 3.13 on Windows. "
                    "Use Python 3.10 or 3.11 for PaddleOCR on Windows."
                )
            from paddleocr import PaddleOCR  # type: ignore
            self._engine = PaddleOCR(
                use_angle_cls=True,
                lang=getattr(settings, "PADDLE_OCR_LANG", "ar")
                # NOTE: do NOT pass show_log here; some versions don't support it
            )
        except Exception as e:
            logger.exception("PaddleOCR initialization failed")
            self._engine = None
            raise RuntimeError(
                "PaddleOCR initialization failed. Ensure a supported Python version "
                "(3.10/3.11 on Windows) and install a compatible paddlepaddle/paddleocr."
            ) from e


    async def _extract_text_from_image(self, image: Image.Image) -> str:
        logger.info("Using PaddleOCRProcessor")
        self._ensure_engine()

        # Convert PIL Image to NumPy array (stable input format)
        import numpy as np
        img_array = np.array(image.convert("RGB"))

        # Call OCR without cls parameter
        result = await asyncio.to_thread(self._engine.ocr, img_array)  # type: ignore

        logger.info(f"Result type: {type(result)}")
        
        items: List[Tuple[List[List[int]], str, float]] = []

        def _arabic_heavy(s: str) -> bool:
            # proportion of Arabic letters among all letters
            letters = [ch for ch in s if ch.isalpha()]
            if not letters:
                return False
            ar = sum(1 for ch in letters if '\u0600' <= ch <= '\u06FF')
            return ar / len(letters) >= 0.6

        def _fix_paddle_arabic_token(s: str) -> str:
            # Some PaddleOCR builds output Arabic tokens in LTR order – flip once
            return s[::-1] if _arabic_heavy(s) else s

        # Apply just before calling rebuild_text_from_boxes(...)
        items = [(bbox, _fix_paddle_arabic_token(text), conf) for (bbox, text, conf) in items]
        
        # Handle dict-based result format (newer PaddleOCR versions)
        if isinstance(result, dict):
            logger.info(f"Dict keys: {list(result.keys())}")
            result_dict = result
            
            texts = (result_dict.get('rec_texts') or 
                    result_dict.get('texts') or [])
            scores = (result_dict.get('rec_scores') or 
                    result_dict.get('scores') or [])
            polys = (result_dict.get('rec_polys') or 
                    result_dict.get('polys') or [])
            
            logger.info(f"Found {len(texts)} texts, {len(scores)} scores, {len(polys)} polys")
            
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
        
        # Handle list format
        elif isinstance(result, list):
            logger.info(f"List format with {len(result)} pages")
            
            # DEBUG: Show structure
            if result:
                first_page = result[0]
                logger.info(f"First page type: {type(first_page)}")
                logger.info(f"First page is dict: {isinstance(first_page, dict)}")
                
                # If first page is a dict, it's the new format!
                if isinstance(first_page, dict):
                    logger.info(f"First page keys: {list(first_page.keys())}")
                    
                    # Extract from dict format
                    texts = first_page.get('rec_texts', [])
                    scores = first_page.get('rec_scores', [])
                    polys = first_page.get('rec_polys', [])
                    
                    logger.info(f"Found {len(texts)} texts in dict")
                    
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
                
                # Classic nested list format
                elif isinstance(first_page, list):
                    logger.info(f"First page length: {len(first_page)}")
                    if first_page:
                        logger.info(f"First detection: {first_page[0]}")
                    
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
        
        logger.info(f"Total extracted: {len(items)} text items from PaddleOCR")
        
        if not items:
            logger.error("No items extracted!")
            logger.error(f"Result structure: {result}")
        
        items = [(bbox, _flip_if_needed(text), conf) for (bbox, text, conf) in items]
        text = rebuild_text_from_boxes(items)
        return text