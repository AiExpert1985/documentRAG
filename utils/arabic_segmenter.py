# utils/arabic_segmenter.py
import re
from typing import List, Dict

# Bullets / dashes
_BULLET = r"[\u2022\u2023\u25E6\u2043•▪\-\–\—\*]"
# Arabic or Latin digits
_DIGIT = r"[0-9٠-٩]"
# Arabic letters
_AR_LET = r"[اأإآبتثجحخدذرزسشصضطظعغفقكلمنهوىي]"

# Item starts (line-begin bullets / numbers / lettered)
_ITEM_LINE_START = re.compile(
    rf"(?m)^(?:{_BULLET}\s+|{_DIGIT}+[)\.\-–]\s+|{_AR_LET}\s*[-–]\s+)"
)

# Inline headers like "عنوان قصير: الوصف..." (colon within first ~60 chars)
_INLINE_HEADER = re.compile(r"(?m)^(.{2,60}):\s+\S")

# A simple “title” heuristic for page headers (no punctuation, relatively short)
_HEADER_LINE = re.compile(r"(?m)^(?=.{4,120}$)(?!.*[\.!\?؟؛:]).+")

def _normalize(text: str) -> str:
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

def segment_arabic(text: str) -> List[Dict[str, str]]:
    """
    Returns: List of {"text": chunk, "kind": "item"|"header"|"paragraph"}.
    Structure-first: recognizes bullets, numbered/lettered items, and inline headers.
    """
    t = _normalize(text)
    if not t:
        return []

    lines = t.split("\n")

    # Optional page header: first non-empty line looks like a title and the next
    # few lines contain items. Emit it as a separate 'header'.
    out: List[Dict[str, str]] = []
    i = 0
    while i < len(lines) and not lines[i].strip():
        i += 1

    def looks_header(idx: int) -> bool:
        return 0 <= idx < len(lines) and _HEADER_LINE.match(lines[idx].strip()) is not None

    def upcoming_contains_items(start: int, horizon: int = 6) -> bool:
        end = min(len(lines), start + horizon)
        block = "\n".join(lines[start:end])
        return (_ITEM_LINE_START.search(block) is not None) or (_INLINE_HEADER.search(block) is not None)

    if looks_header(i) and upcoming_contains_items(i + 1):
        out.append({"text": lines[i].strip(), "kind": "header"})
        i += 1
        # skip a single blank line after header, if present
        if i < len(lines) and not lines[i].strip():
            i += 1

    # Collect items / paragraphs
    buf: List[str] = []
    buf_kind = "paragraph"

    def flush():
        nonlocal buf, buf_kind
        if buf:
            text_block = "\n".join(x for x in buf).strip()
            if text_block:
                out.append({"text": text_block, "kind": buf_kind})
        buf = []
        # default kind for new buffer
        buf_kind = "paragraph"

    def is_item_start(ln: str) -> bool:
        s = ln.strip()
        return bool(_ITEM_LINE_START.match(s) or _INLINE_HEADER.match(s))

    while i < len(lines):
        ln = lines[i]
        if not ln.strip():
            # blank line: paragraph boundary
            flush()
            i += 1
            continue

        if is_item_start(ln):
            # start of a new item
            flush()
            buf_kind = "item"
            buf.append(ln)
            i += 1
            # attach following non-empty lines until next item start or blank
            while i < len(lines) and lines[i].strip() and not is_item_start(lines[i]):
                buf.append(lines[i])
                i += 1
            flush()
            continue

        # normal paragraph continuation
        buf.append(ln)
        i += 1

    flush()
    return out
