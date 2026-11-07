"""Parse a lithology legend: find the swatch boxes, read their names, describe their patterns."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import pytesseract
from PIL import Image

import difflib

from .features import describe, row_stats
from .shape import shape_descriptor

VOCAB = [
    "Limestone", "Dolomite", "Anhydrite", "Halite", "Salt", "Gypsum", "Coal", "Lignite", "Clay", "Claystone",
    "Shale", "Mudstone", "Marl", "Siltstone", "Silty", "Sand", "Sandstone", "and", "Conglomerate", "Breccia",
    "Chert", "Tuff", "Volcanic", "Igneous", "Cement", "Calcareous", "Dolomitic", "Argillaceous", "Others",
]


@dataclass
class Swatch:
    name: str
    box: tuple[int, int, int, int]  # absolute xyxy
    descriptor: np.ndarray  # window-stat descriptor (same space as column rows)
    shape: np.ndarray  # scale-free stroke-shape descriptor, used for matching
    is_white: bool


def _nonwhite(rgb: np.ndarray) -> np.ndarray:
    rgb = rgb.astype(np.int16)
    return (rgb.min(2) < 200) | (rgb.max(2) - rgb.min(2) >= 30)


def _rects(mask: np.ndarray, close_px: int = 3) -> list[tuple[int, int, int, int]]:
    """Every rectangle whose outline is drawn (>=85% of its perimeter is non-white).

    This deliberately over-generates (frames, brick cells inside a swatch, ...);
    the label-driven matching below picks the right one.
    """
    H, W = mask.shape
    m = (mask * 255).astype(np.uint8)
    if close_px > 1:  # bridge small breaks in scanned outlines
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((close_px, close_px), np.uint8))
        mask = m > 0
    cnts, _ = cv2.findContours(m, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    seen, out = set(), []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w < 12 or h < 8 or w * h > 0.25 * W * H or not (0.3 <= w / h <= 5.0):
            continue
        key = (x // 3, y // 3, w // 3, h // 3)
        if key in seen:
            continue
        seen.add(key)
        # perimeter coverage, allowing the outline to be 2 px thick / off by one
        x1, y1 = x + w - 1, y + h - 1
        top = mask[y : y + 2, x : x1 + 1].any(0)
        bot = mask[max(y1 - 1, 0) : y1 + 1, x : x1 + 1].any(0)
        lef = mask[y : y1 + 1, x : x + 2].any(1)
        rig = mask[y : y1 + 1, max(x1 - 1, 0) : x1 + 1].any(1)
        cov = (top.sum() + bot.sum() + lef.sum() + rig.sum()) / (2 * w + 2 * h)
        if cov >= 0.85:
            out.append((x, y, x + w, y + h))
    return out


def _text_lines(rgb: np.ndarray, scale: float) -> list[tuple[str, tuple[float, float, float, float]]]:
    pil = Image.fromarray(rgb)
    if scale != 1:
        pil = pil.resize((int(pil.width * scale), int(pil.height * scale)), Image.LANCZOS)
    d = pytesseract.image_to_data(pil, config="--psm 11", output_type=pytesseract.Output.DICT)
    words = []
    for i, t in enumerate(d["text"]):
        t = t.strip()
        if t and float(d["conf"][i]) > 30 and any(ch.isalpha() for ch in t):
            l, tp, w, h = (d[k][i] / scale for k in ("left", "top", "width", "height"))
            words.append([t, l, tp, l + w, tp + h])
    # Join words into lines: similar vertical centre and small horizontal gap.
    words.sort(key=lambda w: (round((w[2] + w[4]) / 2 / 6), w[1]))
    lines: list[list] = []
    for w in words:
        if lines:
            L = lines[-1]
            hgt = max(w[4] - w[2], L[4] - L[2])
            if abs((w[2] + w[4]) / 2 - (L[2] + L[4]) / 2) < 0.5 * hgt and 0 <= w[1] - L[3] < 1.5 * hgt:
                L[0] += " " + w[0]
                L[3], L[2], L[4] = w[3], min(L[2], w[2]), max(L[4], w[4])
                continue
        lines.append(list(w))
    return [(l[0], (l[1], l[2], l[3], l[4])) for l in lines]


def _assign_labels(rects, lines) -> dict[tuple, str]:
    """For each text line, find the swatch it labels: the largest drawn rectangle immediately
    left of it, or immediately above it. Several lines can label one swatch ("Silty / Sandstone")."""
    hits: dict[tuple, list[tuple[float, float, str]]] = {}
    for t, (a, b, c, d) in lines:
        if sum(ch.isalpha() for ch in t) < 3:
            continue
        cx, cy, th = (a + c) / 2, (b + d) / 2, d - b
        if any(r[0] <= cx <= r[2] and r[1] <= cy <= r[3] and (r[2] - r[0]) * (r[3] - r[1]) < 0.1 * 1e6 for r in rects
               if (r[2] - r[0]) < 6 * (c - a) + 200):
            continue  # OCR noise from a pattern inside a swatch
        cands = []
        for r in rects:
            x0, y0, x1, y1 = r
            w, h = x1 - x0, y1 - y0
            if x0 <= cx <= x1 and y0 <= cy <= y1:
                continue  # text inside the rectangle: it's a frame, not a swatch
            right = -6 <= a - x1 <= max(1.5 * w, 40) and y0 - 0.5 * th <= cy <= y1 + 0.5 * th
            below = -6 <= b - y1 <= 1.3 * h and x0 - 0.25 * w <= cx <= x1 + 0.25 * w
            if right or below:
                cands.append(((x1 - x0) * (y1 - y0), r))
        if cands:
            r = max(cands)[1]
            hits.setdefault(r, []).append((cy, a, t, th))
    out = {}
    for r, v in hits.items():
        v.sort()  # by vertical centre
        rows: list[list] = []
        for item in v:
            if rows and abs(item[0] - rows[-1][0][0]) < 0.5 * max(item[3], rows[-1][0][3]):
                rows[-1].append(item)
            else:
                rows.append([item])
        out[r] = " ".join(t for row in rows for _, _, t, _ in sorted(row, key=lambda z: z[1]))
    return out


def _clean(name: str) -> str:
    """Snap each OCR'd word to a lithology vocabulary when it is a near-miss ("Slity" -> "Silty")."""
    out = []
    for w in name.split():
        core = w.strip(".,:;")
        if not core.isalpha() or len(core) < 3:  # codes like "Clay-S", "Sand-MSS" are kept verbatim
            out.append(w)
            continue
        m = difflib.get_close_matches(core.capitalize(), VOCAB, n=1, cutoff=0.75)
        out.append(m[0] if m else w)
    return " ".join(out)


def _ocr_strip(rgb: np.ndarray, scale: float) -> str:
    pil = Image.fromarray(rgb)
    pil = pil.resize((int(pil.width * scale * 2), int(pil.height * scale * 2)), Image.LANCZOS)
    t = pytesseract.image_to_string(pil, config="--psm 6").strip()
    return " ".join(w for w in t.split() if sum(ch.isalpha() for ch in w) >= 3)


def parse_legend(page, px_per_ft: float, box=None) -> list[Swatch]:
    x0, y0, x1, y1 = box or page.boxes["lithology_legend"]
    rgb = page.crop((x0, y0, x1, y1))
    scale = max(1.0, 300.0 / page.dpi) if page.dpi else 2.0
    lines = _text_lines(rgb, scale)
    out = []
    close_px = max(3, int(round((page.dpi or 150) / 100)) + 1)
    rects = _rects(_nonwhite(rgb), close_px)
    labelled = _assign_labels(rects, lines)
    # Fallback: rectangles the same size as labelled swatches but with no label → OCR the strip
    # where the label should be (to the right or below, following the labelled ones).
    if labelled:
        ws = np.median([r[2] - r[0] for r in labelled]); hs = np.median([r[3] - r[1] for r in labelled])
        below_layout = np.median([1.0 if (r[3] - r[1]) > 2.5 * 20 else 0.0 for r in labelled]) > 0.5
        for r in rects:
            if r in labelled or abs((r[2] - r[0]) - ws) > 0.15 * ws or abs((r[3] - r[1]) - hs) > 0.15 * hs:
                continue
            if any(abs(r[0] - k[0]) < 5 and abs(r[1] - k[1]) < 5 for k in labelled):
                continue
            a, b, c, d = r
            strip = rgb[d : int(d + 0.9 * hs), a : c] if below_layout else rgb[b:d, c : int(c + 4 * ws)]
            if strip.size:
                t = _ocr_strip(strip, scale)
                if t:
                    labelled[r] = t
    for r, name in labelled.items():
        name = _clean(name)
        a, b, c, d = r
        inset = max(3, int(0.08 * min(c - a, d - b)))
        inner = rgb[b + inset : d - inset, a + inset : c - inset]
        if inner.size == 0:
            continue
        st, _ = row_stats(inner)
        desc = describe(st.sum(0), len(st), inner.shape[1], px_per_ft)
        is_white = bool(desc[0] < 0.01 and desc[2:-2].sum() < 0.02)
        out.append(Swatch(name, (x0 + a, y0 + b, x0 + c, y0 + d), desc, shape_descriptor(inner), is_white))
    return out
