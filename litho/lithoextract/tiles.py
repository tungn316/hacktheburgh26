"""Name pieces of the column by finding the legend's own drawing in them.

Many logs fill the column by repeating the legend's symbol at the same size. There a swatch can be
looked for directly (normalised cross-correlation): its real symbol scores above 0.9 and any other
about 0.5, far clearer than comparing stroke statistics, and it works piece by piece, so it does not
depend on the clustering having kept different rocks apart.

Where the legend draws symbols at another size nothing matches, the page is left to the scale-free
fingerprints in shape.py, and this module changes nothing.
"""
from __future__ import annotations

import cv2
import numpy as np

HIT = 0.7  # a row is explained by a swatch if the drawing matches this well there
WEAK_HIT = 0.6  # second pass, on a page already shown to qualify: a worn or mixed stretch, flagged ambiguous
STRONG = 0.4  # a piece is named from its best swatch's score at or above this
PAGE_MIN = 0.5  # share of the column that must be strongly explained to trust this on a page
MARGIN = 0.10  # best and runner-up closer than this -> ambiguous
MAX_ROWS = 2000  # of a long piece, only this many rows from its middle are searched
MAX_FILL_DIFF = 25  # grey levels: same strokes on a different fill are a different symbol (shale / gray shale)


def _gray(rgb: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32), (0, 0), 1.2)


def _templates(swatches, col_width: int) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """(name, the middle of the swatch's drawing, mean colour). A part well inside the column's width
    survives a scan's small stretch and the marks (P, B) written over the column; a wider one does not."""
    out = []
    for s in swatches:
        t = s.tile
        if s.is_white or t.shape[0] < 8 or t.shape[1] < 16:
            continue
        w = min(t.shape[1], int(0.6 * col_width))
        x = (t.shape[1] - w) // 2
        t = t[:, x : x + w]
        g = _gray(t)
        if g.std() >= 5:  # a flat fill has no drawing to find
            out.append((s.name, g, t.reshape(-1, 3).mean(0)))
    return out


def _score(piece_gray: np.ndarray, piece_rgb: np.ndarray, tmpl: np.ndarray, fill: np.ndarray, hit: float = HIT) -> float:
    """Mean, over the piece's rows, of how well the drawing matches where it covers that row (0 if under hit)."""
    h = tmpl.shape[0]
    if piece_gray.shape[0] < h or piece_gray.shape[1] < tmpl.shape[1]:
        return 0.0
    peak = cv2.matchTemplate(piece_gray, tmpl, cv2.TM_CCOEFF_NORMED).max(axis=1)
    peak = np.where(peak >= hit, peak, 0.0)
    if not peak.any():
        return 0.0
    hit_rows = np.zeros(piece_gray.shape[0], bool)
    cover = np.zeros(piece_gray.shape[0], np.float32)
    for i in np.nonzero(peak)[0]:
        cover[i : i + h] = np.maximum(cover[i : i + h], peak[i])
        hit_rows[i : i + h] = True
    if np.abs(piece_rgb[hit_rows].reshape(-1, 3).mean(0) - fill).max() > MAX_FILL_DIFF:
        return 0.0
    return float(cover.mean())


def name_pieces(rgb: np.ndarray, segs, swatches, twins: dict, commonness, unsure=lambda s: True) -> dict[int, tuple] | None:
    """{index into segs: (name, confidence, alternatives, ambiguous)}, or None if the page doesn't qualify."""
    tmpls = _templates(swatches, rgb.shape[1])
    if len(tmpls) < 2:
        return None
    labels: dict[int, tuple] = {}
    strong_rows = total_rows = 0
    for i, s in enumerate(segs):
        if s.blank:
            continue
        n = s.y1 - s.y0
        total_rows += n
        a = s.y0 + max(0, (n - MAX_ROWS) // 2)
        part = rgb[a : min(s.y1, a + MAX_ROWS)]
        g = _gray(part)
        ranked = sorted(((_score(g, part, t, f), name) for name, t, f in tmpls), reverse=True)
        best, top = ranked[0]
        if best < STRONG:
            continue
        strong_rows += n
        tied = {top} | twins.get(top, set())
        others = [v for v, nm in ranked if nm not in tied]
        second = others[0] if others else 0.0
        name = min(tied, key=lambda m: (commonness(m), m))
        alts = sorted(tied - {name}) + [nm for v, nm in ranked[1:4] if nm not in tied and v > 0]
        labels[i] = (name, round(float((best - second) / best), 3), alts, best - second < MARGIN or len(tied) > 1)
    if not total_rows or strong_rows / total_rows < PAGE_MIN:
        return None
    for i, s in enumerate(segs):
        if s.blank or i in labels:
            continue
        a = s.y0 + max(0, (s.y1 - s.y0 - MAX_ROWS) // 2)
        part = rgb[a : min(s.y1, a + MAX_ROWS)]
        g = _gray(part)
        ranked = sorted(((_score(g, part, t, f, WEAK_HIT), name) for name, t, f in tmpls), reverse=True)
        best, top = ranked[0]
        if best > 0:
            tied = {top} | twins.get(top, set())
            name = min(tied, key=lambda m: (commonness(m), m))
            alts = sorted(tied - {name}) + [nm for v, nm in ranked[1:4] if nm not in tied and v > 0]
            others = [v for v, nm in ranked if nm not in tied]
            labels[i] = (name, round(float((best - (others[0] if others else 0.0)) / best) * 0.5, 3), alts, True)
    # A piece shorter than the drawing can't be judged here. If the fingerprints are sure of it (a thin
    # yellow sand on a compressed log) it keeps their name, otherwise it joins the closer neighbour.
    short = int(np.median([t.shape[0] for _, t, _ in tmpls]))
    for i, s in sorted(enumerate(segs), key=lambda z: z[1].y1 - z[1].y0):
        if s.blank or i in labels or s.y1 - s.y0 >= short or not unsure(s):
            continue
        near = [j for j in (i - 1, i + 1) if 0 <= j < len(segs) and j in labels]
        if near:
            j = min(near, key=lambda q: float(np.linalg.norm(segs[q].descriptor - s.descriptor)))
            labels[i] = labels[j]
    return labels
