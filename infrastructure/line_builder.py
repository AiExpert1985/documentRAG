# infrastructure/line_builder.py
from typing import List, Tuple
import re

BBox = List[List[int]]  # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
_ARABIC_RE = re.compile(r"[\u0600-\u06FF]")

def _cx(b: BBox) -> float: return sum(p[0] for p in b) / 4.0
def _cy(b: BBox) -> float: return sum(p[1] for p in b) / 4.0
def _h(b: BBox) -> float:  ys = [p[1] for p in b]; return max(ys) - min(ys)

def _is_rtl_text(tokens: List[str]) -> bool:
    # Majority Arabic heuristic
    ar = sum(1 for t in tokens if _ARABIC_RE.search(t))
    la = sum(1 for t in tokens if re.search(r"[A-Za-z]", t))
    return ar >= la and ar > 0

def rebuild_text_from_boxes(
    items: List[Tuple[BBox, str, float]],
    *,
    alpha_line: float = 0.7,  # line y-tolerance × median height
    beta_blank: float = 1.6   # blank line gap × median height
) -> str:
    if not items:
        return ""

    # sort by y, then x to get a stable reading pass
    items_sorted = sorted(items, key=lambda it: (_cy(it[0]), _cx(it[0])))
    heights = [_h(b) for b,_,_ in items_sorted if _h(b) > 0]
    Hmed = sorted(heights)[len(heights)//2] if heights else 16.0
    if Hmed <= 0: Hmed = 16.0

    lines: List[List[Tuple[BBox, str, float]]] = []
    cur: List[Tuple[BBox, str, float]] = []
    prev_y = None
    prev_bottom = None

    def flush():
        nonlocal cur
        if not cur:
            return
        # decide RTL/LTR per line using tokens in this line
        tokens = [t for _,t,_ in cur if t]
        rtl = _is_rtl_text(tokens)
        # order tokens by x: RTL -> right-to-left (desc), LTR -> left-to-right (asc)
        cur.sort(key=lambda it: _cx(it[0]), reverse=rtl)
        lines.append(cur.copy())
        cur.clear()

    for b,t,c in items_sorted:
        if not t:
            continue
        cy = _cy(b)
        top = min(p[1] for p in b)
        bottom = max(p[1] for p in b)

        if prev_y is None or abs(cy - prev_y) <= alpha_line * Hmed:
            cur.append((b,t,c))
        else:
            gap = (top - (prev_bottom if prev_bottom is not None else top))
            flush()
            if gap > beta_blank * Hmed:
                lines.append([])  # blank line marker
            cur.append((b,t,c))
        prev_y = cy
        prev_bottom = bottom

    flush()

    out_lines: List[str] = []
    for ln in lines:
        if not ln:
            out_lines.append("")  # blank line
        else:
            out_lines.append(" ".join(tok for _,tok,_ in ln))
    return "\n".join(out_lines).strip()
