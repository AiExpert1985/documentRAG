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
import re
from typing import Callable, List, Dict, Any, Optional, Tuple
from difflib import SequenceMatcher

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

def _normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def _text_overlap_ratio(a: str, b: str) -> float:
    """Cheap, language-agnostic overlap (char-level)."""
    a = _normalize_spaces(a)
    b = _normalize_spaces(b)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _axis_aligned_from_quad(poly):
    xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    return (int(x1), int(y1), int(x2 - x1), int(y2 - y1))

LineRec = Dict[str, Any]
SegmentRec = Dict[str, Any]
BBoxPx  = Tuple[int, int, int, int]


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
    async def _extract_lines(self, image: Image.Image) -> List[LineRec]:
        """
        Extract lines with bounding boxes from image.
        
        Returns:
            List of dicts with keys:
            - line_id: str
            - poly: List[List[int]] (4-point polygon)
            - bbox_px: Tuple[int, int, int, int] (x, y, w, h)
            - text: str
            - conf: float (0-1)
        """
        raise NotImplementedError

    @abstractmethod
    async def _extract_text_from_image(self, image: Image.Image) -> str:
        """Engine-specific page OCR -> plain text with reliable newlines."""
        raise NotImplementedError

    async def process(
        self,
        file_path: str,
        file_type: str,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Tuple[List[DocumentChunk], Dict[int, Dict[str, Any]]]:
        """
        Process document and return (chunks, geometry_by_page).
        
        Returns:
            Tuple of:
            - List[DocumentChunk]: Text chunks for embedding/search
            - Dict[int, Dict]: Geometry data per page for highlighting
            {
                page_index: {
                    "lines": List[LineRec],
                    "segments": List[SegmentRec]
                }
            }
        """
        # 1) Load page images
        images = await self.load_images(file_path, file_type)
        total_pages = len(images)
        
        all_chunks: List[DocumentChunk] = []
        geometry_by_page: Dict[int, Dict[str, Any]] = {}
        
        # 2) Process each page
        for page_idx, image in enumerate(images, start=1):
            self._update_progress(progress_callback, page_idx, total_pages)
            
            try:
                # Extract lines with bounding boxes
                lines = await self._extract_lines(image)
                
                if not lines:
                    logger.warning(f"No lines extracted from page {page_idx}")
                    continue
                
                # Segment into paragraphs using arabic_segmenter
                segments = await self._segment_paragraphs(lines, page_idx)
                
                if not segments:
                    logger.warning(f"No segments created for page {page_idx}")
                    continue
                
                # Convert segments to chunks
                page_chunks = self._segments_to_chunks(segments, file_path, page_idx)
                all_chunks.extend(page_chunks)
                
                # Store geometry for this page
                geometry_by_page[page_idx] = {
                    "lines": lines,
                    "segments": segments
                }
                
            except Exception as e:
                logger.error(f"Failed processing page {page_idx}: {e}")
                # Continue to next page instead of failing entire document
                continue
        
        if not all_chunks:
            raise DocumentProcessingError(
                "No text extracted from document",
                ErrorCode.NO_TEXT_FOUND
            )
        
        logger.info(
            f"Processed {len(images)} pages into {len(all_chunks)} chunks "
            f"with geometry for {len(geometry_by_page)} pages"
        )
        
        return all_chunks, geometry_by_page


    def _segments_to_chunks(
        self,
        segments: List[SegmentRec],
        source: str,
        page_index: int
    ) -> List[DocumentChunk]:
        """
        Convert segments to DocumentChunks with metadata links.
        
        Uses RecursiveCharacterTextSplitter for length-based splitting
        while preserving segment_id references.
        """
        # Build LangchainDocuments from segments
        docs: List[LangchainDocument] = []
        for seg in segments:
            docs.append(
                LangchainDocument(
                    page_content=seg["text"],
                    metadata={
                        "page": page_index,
                        "source": source,
                        "segment_kind": seg.get("type", "paragraph"),
                        "segment_id": seg["segment_id"]  # Critical for highlighting
                    }
                )
            )
        
        # Split long segments
        split_docs = self.text_splitter.split_documents(docs)
        
        # Merge tiny paragraphs (respects atomic boundaries)
        merged_docs = self._merge_tiny_paragraphs(split_docs)
        
        # Build final chunks
        chunks: List[DocumentChunk] = []
        for i, doc in enumerate(merged_docs):
            # Collect segment_ids (chunk may span multiple segments after splitting/merging)
            segment_ids = []
            if "segment_id" in doc.metadata:
                segment_ids.append(doc.metadata["segment_id"])
            
            # Extract segment and line references from source document
            segment_ids = []
            if "segment_id" in doc.metadata:
                segment_ids.append(doc.metadata["segment_id"])

            line_ids = []
            if "line_ids" in doc.metadata:
                val = doc.metadata["line_ids"]
                if isinstance(val, list):
                    line_ids.extend(val)

            chunks.append(
                DocumentChunk(
                    id=f"{uuid.uuid4()}_{i}",
                    content=doc.page_content,
                    document_id="",
                    metadata={
                        "page": doc.metadata.get("page", page_index),
                        "source": doc.metadata.get("source", source),
                        "segment_kind": doc.metadata.get("segment_kind", "paragraph"),
                        "segment_ids": segment_ids,      # ADDED
                        "line_ids": line_ids,            # ADDED
                    }
                )
            )
        
        return chunks

    async def load_images(self, file_path: str, file_type: str) -> List[Image.Image]:
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

    # def _build_chunks(self, docs: List[LangchainDocument]) -> List[DocumentChunk]:
    #     """Build final DocumentChunk objects (clean content, no display formatting)."""
    #     chunks: List[DocumentChunk] = []
    #     for i, doc in enumerate(docs):
    #         chunks.append(
    #             DocumentChunk(
    #                 id=f"{uuid.uuid4()}_{i}",
    #                 content=doc.page_content,  # Clean content
    #                 document_id="",
    #                 metadata={
    #                     "page": (doc.metadata or {}).get("page", -1),
    #                     "source": (doc.metadata or {}).get("source", ""),
    #                     "segment_kind": (doc.metadata or {}).get("segment_kind", "paragraph")
    #                 }
    #             )
    #         )
    #     return chunks
    
    async def _segment_paragraphs(self, lines: List[LineRec], page_index: int) -> List[SegmentRec]:
        """Segment paragraphs with safety net for oversized segments."""
        
        # 1) Rebuild page text
        page_text = "\n".join(_normalize_spaces(ln["text"]) for ln in lines if ln.get("text"))
        if not page_text:
            return []

        # 2) Call segmenter
        from utils.arabic_segmenter import segment_arabic
        segments = segment_arabic(page_text) or []

        # 3) Map segments to line_ids
        results: List[SegmentRec] = []
        for seg in segments:
            seg_text = _normalize_spaces(seg.get("text", ""))
            if not seg_text:
                continue

            # Find matching lines
            candidate_lines = []
            for ln in lines:
                ln_text = _normalize_spaces(ln.get("text", ""))
                if not ln_text:
                    continue
                
                if ln_text in seg_text or seg_text in ln_text or _text_overlap_ratio(ln_text, seg_text) >= 0.65:
                    candidate_lines.append(ln)
            
            if not candidate_lines:
                continue

            line_ids = [ln["line_id"] for ln in candidate_lines]
            
            # NEW: Safety check - split if segment has too many lines
            MAX_LINES_PER_SEGMENT = 8
            if len(line_ids) > MAX_LINES_PER_SEGMENT:
                logger.warning(
                    f"Page {page_index}: Segment has {len(line_ids)} lines (>{MAX_LINES_PER_SEGMENT}). "
                    f"Splitting into smaller chunks. Check arabic_segmenter logic."
                )
                
                # Split into chunks of MAX_LINES_PER_SEGMENT
                for i in range(0, len(line_ids), MAX_LINES_PER_SEGMENT):
                    chunk_line_ids = line_ids[i:i + MAX_LINES_PER_SEGMENT]
                    results.append({
                        "segment_id": f"seg_{uuid.uuid4().hex[:8]}",
                        "line_ids": chunk_line_ids,
                        "text": seg_text,  # Original text (chunking will handle excerpts)
                        "type": seg.get("kind", "paragraph"),
                        "page_index": page_index
                    })
            else:
                results.append({
                    "segment_id": f"seg_{uuid.uuid4().hex[:8]}",
                    "line_ids": line_ids,
                    "text": seg_text,
                    "type": seg.get("kind", "paragraph"),
                    "page_index": page_index
                })

        # MVP guard: if segmenter returned nothing, treat each line as segment
        if not results and lines:
            for ln in lines:
                if not ln.get("text"):
                    continue
                results.append({
                    "segment_id": f"seg_{uuid.uuid4().hex[:8]}",
                    "line_ids": [ln["line_id"]],
                    "text": _normalize_spaces(ln["text"]),
                    "type": "line",
                    "page_index": page_index
                })

        return results

# -----------------------------
# Tesseract
# -----------------------------
class TesseractProcessor(BaseOCRProcessor):
    """
    Tesseract-backed OCR that returns line-level geometry.
    Uses pytesseract.image_to_data to capture bboxes.
    """

    def __init__(self, lang: str = "ara+eng", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lang = lang

    async def _extract_lines(self, image: Image.Image) -> List[LineRec]:
        """
        Extract lines with geometry from a PIL image using Tesseract.
        Returns a unified list of dicts:
          {
            "line_id": "ln_xxx",
            "poly": [[x1,y1],[x2,y1],[x2,y2],[x1,y2]],   # pixel-space quad
            "bbox_px": (x, y, w, h),                     # pixel-space AABB
            "text": "line text",
            "conf": 0.0..1.0                             # min word conf in the line
          }
        """
        from pytesseract import image_to_data, Output  # local import to avoid hard dep if not enabled

        # Make sure we pass RGB
        rgb = image.convert("RGB")

        data = image_to_data(rgb, lang=self._lang, output_type=Output.DICT)

        n = len(data["text"])
        if n == 0:
            return []

        # Group words into lines by (block_num, line_num)
        by_line: Dict[Tuple[int, int], List[int]] = {}
        for i in range(n):
            # Tesseract uses "-1" for non-words/metadata; skip those
            try:
                conf_i = float(data["conf"][i])
            except Exception:
                conf_i = -1.0
            if conf_i < 0:
                continue

            blk = int(data.get("block_num", [0])[i] or 0)
            ln  = int(data.get("line_num",  [0])[i] or 0)
            by_line.setdefault((blk, ln), []).append(i)

        lines: List[LineRec] = []

        for (blk, ln), idxs in by_line.items():
            xs: List[int] = []
            ys: List[int] = []
            x2s: List[int] = []
            y2s: List[int] = []
            parts: List[str] = []
            min_conf = 1000.0

            for i in idxs:
                l = int(data["left"][i]);  t = int(data["top"][i])
                w = int(data["width"][i]); h = int(data["height"][i])

                xs.append(l); ys.append(t); x2s.append(l + w); y2s.append(t + h)

                txt = (data["text"][i] or "").strip()
                if txt:
                    parts.append(txt)

                try:
                    c = float(data["conf"][i])
                    if c < min_conf:
                        min_conf = c
                except Exception:
                    # ignore malformed conf entries
                    pass

            if not xs or not parts:
                continue

            x1, y1, x2, y2 = min(xs), min(ys), max(x2s), max(y2s)
            bbox_px: BBoxPx = (x1, y1, x2 - x1, y2 - y1)

            lines.append({
                "line_id": f"ln_{uuid.uuid4().hex[:8]}",
                "poly": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                "bbox_px": bbox_px,
                "text": " ".join(parts),
                "conf": 0.0 if min_conf == 1000.0 else max(0.0, min(1.0, min_conf / 100.0)),
            })

        return lines
    
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reader = None

    async def _get_reader(self):
        if self.reader is not None:
            return
        try:
            import easyocr  # type: ignore
        except Exception as e:
            raise RuntimeError(f"EasyOCR not available: {e}")
        # languages: adjust to your corpus
        self.reader = easyocr.Reader(['ar', 'en'], gpu=False)

    async def _extract_lines(self, image: Image.Image) -> List[Dict[str, Any]]:
        await self._get_reader()
        # easyocr accepts ndarray; convert
        import numpy as np
        arr = np.array(image.convert("RGB"))
        # detail=1 returns list of [bbox, text, conf]
        reader = self.reader  # Local var
        assert reader is not None
        results = reader.readtext(arr, detail=1, paragraph=False)
        lines: List[Dict[str, Any]] = []
        for bbox, text, conf in results:
            # bbox is [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
            if not text:
                continue
            bbox_px = _axis_aligned_from_quad(bbox)
            lines.append({
                "line_id": f"ln_{uuid.uuid4().hex[:8]}",
                "poly": [[int(x), int(y)] for x, y in bbox],
                "bbox_px": bbox_px,
                "text": str(text),
                "conf": float(conf) if conf is not None else 0.0,
            })
        return lines

    async def _extract_text_from_image(self, image: Image.Image) -> str:
        logger.info("Using EasyOCRProcessor")
        await self._get_reader()
        
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
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ocr = None

    async def _ensure_reader(self):
        if self.ocr is not None:
            return
        try:
            from paddleocr import PaddleOCR  # type: ignore
        except Exception as e:
            raise RuntimeError(f"PaddleOCR not available: {e}")
        # use_angle_cls helps with rotation; adjust langs as needed
        self.ocr = PaddleOCR(use_angle_cls=True, lang='ar')  # or 'ar', 'en', 'ar+en' depending on support

    async def _extract_lines(self, image: Image.Image) -> List[Dict[str, Any]]:
        await self._ensure_reader()
        assert self.ocr is not None  # Add this line
        import numpy as np
        arr = np.array(image.convert("RGB"))
        result = self.ocr.ocr(arr, cls=True)
        lines: List[Dict[str, Any]] = []
        # Paddle returns list per image; first element is results
        for det in (result[0] or []):
            poly, (text, conf) = det
            if not text:
                continue
            bbox_px = _axis_aligned_from_quad(poly)
            lines.append({
                "line_id": f"ln_{uuid.uuid4().hex[:8]}",
                "poly": [[int(x), int(y)] for x, y in poly],
                "bbox_px": bbox_px,
                "text": str(text),
                "conf": float(conf) if conf is not None else 0.0,
            })
        return lines

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