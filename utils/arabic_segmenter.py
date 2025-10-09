# utils/arabic_segmenter.py
"""
Structure-first Arabic text segmenter.

Recognizes document structure before length-based splitting:
- Headers (page titles, section headings)
- List items (bullets, numbered, lettered)
- Paragraphs (regular text blocks)

Preserves semantic boundaries for better chunking downstream.
"""
import re
from typing import List, Dict

# Pattern definitions
_BULLET = r"[\u2022\u2023\u25E6\u2043•▪\-\–\—\*]"  # Bullet characters
_DIGIT = r"[0-9٠-٩]"  # Arabic or Latin digits
_AR_LET = r"[اأإآبتثجحخدذرزسشصضطظعغفقكلمنهوىي]"  # Arabic letters

# Item start patterns (line-begin bullets/numbers/letters)
_ITEM_LINE_START = re.compile(
    rf"(?m)^(?:{_BULLET}\s+|{_DIGIT}+[)\.\-–]\s+|{_AR_LET}\s*[-–]\s+)"
)

# Inline headers with colon (e.g., "عنوان قصير: الوصف...")
_INLINE_HEADER = re.compile(r"(?m)^(.{2,60}):\s+\S")

# Simple title heuristic (short, no punctuation)
_HEADER_LINE = re.compile(r"(?m)^(?=.{4,120}$)(?!.*[\.!\?؟؛:]).+")


def _normalize(text: str) -> str:
    """Normalize line breaks and whitespace."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)  # Collapse spaces
    text = re.sub(r"\n{3,}", "\n\n", text)  # Max 2 consecutive newlines
    return text.strip()


def segment_arabic(text: str) -> List[Dict[str, str]]:
    """
    Segment Arabic text by structure (headers, items, paragraphs).
    
    Args:
        text: Raw OCR text with line breaks
    
    Returns:
        List of {"text": content, "kind": "header"|"item"|"paragraph"}
    
    Algorithm:
        1. Detect optional page header (title + upcoming items)
        2. Identify list items (bullets, numbers, letters)
        3. Group remaining text into paragraphs
        4. Preserve blank lines as boundaries
    """
    text = _normalize(text)
    if not text:
        return []

    lines = text.split("\n")
    output: List[Dict[str, str]] = []
    
    # Skip leading blank lines
    i = 0
    while i < len(lines) and not lines[i].strip():
        i += 1

    # Helper functions
    def looks_like_header(idx: int) -> bool:
        """Check if line matches header pattern."""
        if not (0 <= idx < len(lines)):
            return False
        return _HEADER_LINE.match(lines[idx].strip()) is not None

    def upcoming_has_items(start: int, horizon: int = 6) -> bool:
        """Check if upcoming lines contain list items."""
        end = min(len(lines), start + horizon)
        block = "\n".join(lines[start:end])
        return (
            _ITEM_LINE_START.search(block) is not None or 
            _INLINE_HEADER.search(block) is not None
        )

    def is_item_start(line: str) -> bool:
        """Check if line starts a list item."""
        s = line.strip()
        return bool(_ITEM_LINE_START.match(s) or _INLINE_HEADER.match(s))

    # Detect page header
    if looks_like_header(i) and upcoming_has_items(i + 1):
        output.append({"text": lines[i].strip(), "kind": "header"})
        i += 1
        
        # Skip single blank line after header
        if i < len(lines) and not lines[i].strip():
            i += 1

    # Process remaining content
    buffer: List[str] = []
    buffer_kind = "paragraph"

    def flush_buffer():
        """Flush current buffer to output."""
        nonlocal buffer, buffer_kind
        if buffer:
            text_block = "\n".join(buffer).strip()
            if text_block:
                output.append({"text": text_block, "kind": buffer_kind})
        buffer = []
        buffer_kind = "paragraph"

    while i < len(lines):
        line = lines[i]
        
        # Blank line = paragraph boundary
        if not line.strip():
            flush_buffer()
            i += 1
            continue

        # List item
        if is_item_start(line):
            flush_buffer()
            buffer_kind = "item"
            buffer.append(line)
            i += 1
            
            # Attach continuation lines (until next item or blank)
            while i < len(lines) and lines[i].strip() and not is_item_start(lines[i]):
                buffer.append(lines[i])
                i += 1
            
            flush_buffer()
            continue

        # Regular paragraph text
        buffer.append(line)
        i += 1

    flush_buffer()
    return output