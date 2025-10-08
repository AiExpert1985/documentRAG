import re
from typing import List

BULLET = r"[\u2022\u2023\u25E6\u2043•▪\-\–\—\*]"  # common bullet/dash glyphs
AR_NUM = r"[0-9٠-٩]"                               # Arabic + Latin digits
AR_LETTER = r"[اأإآبتثجحخدذرزسشصضطظعغفقكلمنهوىي]"  # Arabic letters

# A list item can start with a bullet/dash, a number + delimiter, or an Arabic letter + dash.
ITEM_START = rf"(?m)^(?:{BULLET}\s+|{AR_NUM}+\)|{AR_NUM}+[\.\-–]\s+|{AR_LETTER}\s*[-–]\s+)"

# Header line ending with colon (Latin or Arabic colon lookalikes)
HEADER_LINE = r"(?m)^[^\n]{2,80}:\s*$"

def normalize(text: str) -> str:
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

def split_blocks(text: str) -> List[str]:
    # first split by blank lines; keeps paragraphs separate
    parts = re.split(r"\n{2,}", text)
    return [p.strip() for p in parts if p.strip()]

def split_items(block: str) -> List[str]:
    # if block looks like a list, split into items; otherwise look for header: bodies
    if re.search(ITEM_START, block):
        # split *before* each item start
        pieces = re.split(rf"(?={ITEM_START})", block)
        return [p.strip() for p in pieces if p.strip()]
    # split on header lines like "عنوان: \nالوصف ..."
    lines = block.split("\n")
    out, cur = [], []
    for ln in lines:
        if re.match(HEADER_LINE, ln) and cur:
            out.append("\n".join(cur).strip()); cur = [ln]
        else:
            cur.append(ln)
    if cur: out.append("\n".join(cur).strip())
    return [p for p in out if p]

def segment_arabic(text: str) -> List[str]:
    text = normalize(text)
    segs: List[str] = []
    for blk in split_blocks(text):
        segs.extend(split_items(blk))
    return segs
