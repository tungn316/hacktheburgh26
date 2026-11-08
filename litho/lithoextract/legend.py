"""Parse a lithology legend: find the swatch boxes, read their names, describe their patterns."""
from __future__ import annotations

from dataclasses import dataclass

import difflib
import re

import cv2
import numpy as np
import pytesseract
from PIL import Image

from .features import classify_pixels, describe, row_stats
from .shape import shape_descriptor

VOCAB = [
    "Limestone", "Dolomite", "Anhydrite", "Halite", "Salt", "Gypsum", "Coal", "Lignite", "Clay", "Claystone",
    "Shale", "Mudstone", "Marl", "Siltstone", "Silt", "Silty", "Sand", "Sandstone", "and", "Conglomerate", "Breccia",
    "Chert", "Tuff", "Volcanic", "Igneous", "Cement", "Calcareous", "Dolomitic", "Argillaceous", "Others",
    "Shaly", "Unknown", "Till", "Bentonite", "Siderite", "Limonite", "Metamorphic", "Marlstone", "Ironstone", "Evaporite",
]


@dataclass
class Swatch:
    name: str
    box: tuple[int, int, int, int]  # absolute xyxy
    descriptor: np.ndarray  # window-stat descriptor (same space as column rows)
    shape: np.ndarray  # scale-free stroke-shape descriptor, used for matching
    is_white: bool
    ink: np.ndarray  # binary ink mask of the swatch interior, for spotting identical symbols


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
        if w < 12 or h < 8 or w * h > 0.25 * W * H or not (0.3 <= w / h <= 12.0):
            continue
        key = (x // 3, y // 3, w // 3, h // 3)
        if key in seen:
            continue
        seen.add(key)
        if _coverage(mask, (x, y, x + w, y + h)) >= 0.85:
            out.append((x, y, x + w, y + h))
    return out


def _coverage(mask: np.ndarray, r) -> float:
    """Share of the rectangle's perimeter that is drawn, allowing the outline to be 2 px thick / off by one."""
    x, y, w, h = r[0], r[1], r[2] - r[0], r[3] - r[1]
    x1, y1 = x + w - 1, y + h - 1
    top = mask[y : y + 2, x : x1 + 1].any(0)
    bot = mask[max(y1 - 1, 0) : y1 + 1, x : x1 + 1].any(0)
    lef = mask[y : y1 + 1, x : x + 2].any(1)
    rig = mask[y : y1 + 1, max(x1 - 1, 0) : x1 + 1].any(1)
    return float((top.sum() + bot.sum() + lef.sum() + rig.sum()) / (2 * w + 2 * h))


def _text_lines(rgb: np.ndarray, scale: float, swatches=()) -> list[tuple[str, tuple[float, float, float, float]]]:
    """OCR'd text lines. ``swatches`` are small drawn boxes: words inside them are pattern noise
    ("i=", "fre]") and are dropped, and a line never continues across one, so the labels of two
    legend columns side by side stay separate."""
    rgb = rgb.copy()
    for a, b, c, d_ in swatches:  # the pattern inside a swatch only ever OCRs as junk glued to the label
        rgb[b:d_, a:c] = 255
    pil = Image.fromarray(rgb)
    if scale != 1:
        pil = pil.resize((int(pil.width * scale), int(pil.height * scale)), Image.LANCZOS)
    d = pytesseract.image_to_data(pil, config="--psm 11", output_type=pytesseract.Output.DICT)
    words = []
    for i, t in enumerate(d["text"]):
        t = t.strip()
        if t and float(d["conf"][i]) > 30 and any(ch.isalpha() for ch in t):
            l, tp, w, h = (d[k][i] / scale for k in ("left", "top", "width", "height"))
            cx, cy = l + w / 2, tp + h / 2
            if any(a <= cx <= c and b <= cy <= d for a, b, c, d in swatches):
                continue
            words.append([t, l, tp, l + w, tp + h, float(d['conf'][i])])

    def box_between(L, w) -> bool:  # a label never spans a swatch (its middle lies inside the joined line)
        return any(L[1] <= (a + c) / 2 <= w[3] and b < max(L[4], w[4]) and d > min(L[2], w[2])
                   for a, b, c, d in swatches)

    # Join words into lines: similar vertical centre and small horizontal gap.
    words.sort(key=lambda w: (round((w[2] + w[4]) / 2 / 6), w[1]))
    lines: list[list] = []
    for w in words:
        if lines:
            L = lines[-1]
            hgt = max(w[4] - w[2], L[4] - L[2])
            if abs((w[2] + w[4]) / 2 - (L[2] + L[4]) / 2) < 0.5 * hgt and 0 <= w[1] - L[3] < 1.5 * hgt \
                    and not box_between(L, w):
                L[0] += " " + w[0]
                L[3], L[2], L[4] = w[3], min(L[2], w[2]), max(L[4], w[4])
                L[5].append(w)
                continue
        lines.append(list(w[:5]) + [[w]])
    return [_strip_junk(L) for L in lines]


def _strip_junk(L) -> tuple[str, tuple[float, float, float, float]]:
    """Leading 1-2 letter "words" are a borderless symbol read as text ("oO BRECCIA", "HB COAL")."""
    ws = L[5]

    def junk(w, rest) -> bool:
        n_alpha = sum(ch.isalpha() for ch in w[0])
        if n_alpha <= 2:
            return True
        # "Per BENTONITE", "Popa ANHYDRITE": a short, unsure, non-rock word in front of a rock name
        return (n_alpha <= 4 and w[5] < 60 and w[0].lower() not in ("and", "or")
                and not _is_rock_word(w[0]) and any(_is_rock_word(r[0]) for r in rest))

    while len(ws) > 1 and junk(ws[0], ws[1:]):
        ws = ws[1:]
    if ws is L[5]:
        return L[0], (L[1], L[2], L[3], L[4])
    return " ".join(w[0] for w in ws), (ws[0][1], min(w[2] for w in ws), ws[-1][3], max(w[4] for w in ws))


def _assign_labels(rects, lines) -> dict[tuple, str]:
    """For each text line, find the swatch it labels: the largest drawn rectangle immediately
    left of it, or immediately above it. Several lines can label one swatch ("Silty / Sandstone")."""
    hits: dict[tuple, list[tuple[float, float, str]]] = {}
    picks = []
    for t, (a, b, c, d) in lines:
        if sum(ch.isalpha() for ch in t) < 3 and t.lower() not in ("or", "and", "&"):
            continue
        cx, cy, th = (a + c) / 2, (b + d) / 2, d - b
        if any(r[0] <= cx <= r[2] and r[1] <= cy <= r[3] and (r[2] - r[0]) * (r[3] - r[1]) < 0.1 * 1e6 for r in rects
               if (r[2] - r[0]) < 6 * (c - a) + 200 and r[3] - r[1] > 1.3 * th):  # smaller: a letter's outline
            continue  # OCR noise from a pattern inside a swatch
        cands = []
        for r in rects:
            x0, y0, x1, y1 = r
            w, h = x1 - x0, y1 - y0
            if x0 <= cx <= x1 and y0 <= cy <= y1:
                continue  # text inside the rectangle: it's a frame, not a swatch
            right = -6 <= a - x1 <= max(1.5 * w, 40) and y0 - 0.5 * th <= cy <= y1 + 0.5 * th
            below = -6 <= b - y1 <= 1.3 * h and x0 <= cx <= x1  # under the box, not the next row's label
            if right or below:
                gap = max(0.0, a - x1) if right else max(0.0, b - y1)
                cands.append((round(gap / 4), not right, -(x1 - x0) * (y1 - y0), r))
        if cands:
            picks.append((min(cands), cands, (cy, a, t, th)))
    # Legends put labels either beside or under their boxes. When nearly every label is beside its
    # box, a box "above" a label is the previous row's swatch (e.g. above a borderless coal icon).
    n_right = sum(not best[1] for best, _, _ in picks)
    beside_only = n_right >= 3 and n_right >= 0.8 * len(picks)
    for best, cands, item in picks:
        if beside_only:
            cands = [c for c in cands if not c[1]]
            if not cands:
                continue
            best = min(cands)
        hits.setdefault(best[3], []).append(item)  # the nearest box; on a tie left beats above, then largest
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
    words = name.split()
    # OCR sometimes splits a word ("Anhyd rite"): rejoin a pair when only the joined form is a rock word.
    i = 0
    while i < len(words) - 1:
        a, b = words[i], words[i + 1]
        if a.isalpha() and b.isalpha() and not _is_rock_word(a) and not _is_rock_word(b) and _is_rock_word(a + b):
            words[i : i + 2] = [a + b.lower()]
        i += 1
    out = []
    for w in words:
        core = w.strip(".,:;")
        if not core.isalpha() or len(core) < 3:  # codes like "Clay-S", "Sand-MSS" are kept verbatim
            out.append(w)
            continue
        m = difflib.get_close_matches(core.capitalize(), VOCAB, n=1, cutoff=0.75)
        out.append(m[0] if m else w)
    return " ".join(out)


def _untitle(name: str) -> str:
    """ALL-CAPS legends ("SHALE GRAY") -> "Shale Gray"; short codes ("MSS", "LS") stay as they are."""
    if name.upper() != name:
        return name
    return " ".join(w.capitalize() if w.isalpha() and len(w) >= 4 else w.lower() if w.lower() in ("or", "and") else w
                    for w in name.split())


def _grid_holes(small: list, nw: np.ndarray, tol: int = 4) -> list[tuple[int, int, int, int]]:
    """Swatch-sized boxes missing from a legend grid.

    Columns are >= 2 boxes sharing left and right edges (at least 3 boxes over all columns); rows
    are box tops shared by >= 2 boxes, plus any row missing from an otherwise regular spacing.
    Where a column meets a row with no box there, and there is ink, a box is assumed.
    """
    cols: list[list] = []
    for r in sorted(small):
        for c in cols:
            if abs(c[0][0] - r[0]) <= tol and abs(c[0][2] - r[2]) <= tol:
                c.append(r)
                break
        else:
            cols.append([r])
    cols = [c for c in cols if len(c) >= 2]
    if len(cols) < 2 or max(len(c) for c in cols) < 3:
        return []
    tops = sorted(r[1] for c in cols for r in c)
    rows: list[list[int]] = []
    for y in tops:
        if rows and y - rows[-1][-1] <= tol:
            rows[-1].append(y)
        else:
            rows.append([y])
    rows = [int(np.median(g)) for g in rows if len(g) >= 2]
    if len(rows) >= 3:  # fill rows skipped by a regular pitch (a row where only one box was traced)
        pitch = float(np.median(np.diff(rows)))
        filled = [rows[0]]
        for y in rows[1:]:
            k = round((y - filled[-1]) / pitch)
            if k >= 2 and abs((y - filled[-1]) - k * pitch) <= tol * k:
                filled += [round(filled[-1] + i * (y - filled[-1]) / k) for i in range(1, k)]
            filled.append(y)
        rows = filled
    out = []
    for c in cols:
        x0, x1 = int(np.median([r[0] for r in c])), int(np.median([r[2] for r in c]))
        h = int(np.median([r[3] - r[1] for r in c]))
        for y in rows:
            if any(abs(r[1] - y) <= tol for r in c):
                continue
            box = (x0, y, x1, min(nw.shape[0], y + h))
            if any(not (box[2] <= q[0] or q[2] <= box[0] or box[3] <= q[1] or q[3] <= box[1]) for q in small):
                continue
            inner = nw[box[1] + 2 : box[3] - 2, box[0] + 2 : box[2] - 2]
            if inner.size and inner.mean() > 0.03:
                out.append(box)
    return out


def _is_rock_word(text: str) -> bool:
    return any(difflib.get_close_matches(w.strip(".,:;").capitalize(), VOCAB, n=1, cutoff=0.8)
               for w in text.split() if len(w) >= 3 and w.lower() not in ("and", "or"))


def _borderless(nw: np.ndarray, lines, labelled: dict) -> dict[tuple, str]:
    """Symbols drawn with no box (chert triangles, a coal square, scattered breccia clasts...).

    A text line that names a rock but was matched to no box gets the ink immediately to its left,
    on its own row, as its swatch: as tall as the labelled swatches, and as wide as that ink runs
    (gaps narrower than ~1/4 swatch width are bridged, so sparse symbols stay in one piece).
    """
    if labelled:
        ws = float(np.median([r[2] - r[0] for r in labelled]))
        hs = float(np.median([r[3] - r[1] for r in labelled]))
    else:
        ws = hs = 0.0
    used = {t for t in labelled.values()}
    taken = [r for r in labelled]
    ths = [d - b for t, (a, b, c, d) in lines if sum(ch.isalpha() for ch in t) >= 3]
    th_med = float(np.median(ths)) if ths else 0.0
    out = {}
    for t, (a, b, c, d) in lines:
        th = d - b
        if t in used or th > 1.6 * th_med or sum(ch.isalpha() for ch in t) < 3 or not _is_rock_word(t):
            continue
        cy = (b + d) / 2
        h = max(hs, 1.4 * th)
        y0, y1 = int(max(0, cy - h / 2)), int(min(nw.shape[0], cy + h / 2))
        reach = int(1.4 * ws) if ws else int(12 * th)
        band = nw[y0:y1, max(0, int(a) - reach) : max(0, int(a) - 2)]
        if band.size == 0:
            continue
        cols = band.any(0)[::-1]  # walking left from the label
        bridge = max(4, int(0.25 * ws)) if ws else int(2 * th)
        start = next((i for i, v in enumerate(cols[: max(bridge, int(1.5 * th))]) if v), None)
        if start is None:
            continue
        end, gap = start, 0
        for i in range(start, len(cols)):
            if cols[i]:
                end, gap = i, 0
            else:
                gap += 1
                if gap > bridge:
                    break
        right = int(a) - 2 - start
        left = int(a) - 2 - end - 1
        box = (max(0, left), y0, right, y1)
        if box[2] - box[0] < 6 or nw[box[1] : box[3], box[0] : box[2]].sum() < 15:
            continue
        if any(not (box[2] <= q[0] or q[2] <= box[0] or box[3] <= q[1] or q[3] <= box[1]) for q in taken):
            continue  # that ink already belongs to a boxed swatch
        out[box] = t
        taken.append(box)
    return out


def canonical_name(name: str) -> str:
    """Collapse "X and Xy" legend names onto the specific rock: "Sand and Sandstone" -> "Sandstone".

    Only when one half is a prefix of the other, so genuine mixtures ("Shale and Clay") are kept.
    """
    parts = [p.strip() for p in re.split(r"\s+(?:and|&)\s+|\s*/\s*", name.strip(), flags=re.I)]
    if len(parts) == 2 and all(parts):
        a, b = sorted(parts, key=len)
        if b.lower().startswith(a.lower()):
            return b
    return name.strip()


def _ocr_strip(rgb: np.ndarray, scale: float) -> str:
    pil = Image.fromarray(rgb)
    pil = pil.resize((int(pil.width * scale * 2), int(pil.height * scale * 2)), Image.LANCZOS)
    t = pytesseract.image_to_string(pil, config="--psm 6").strip()
    return " ".join(w for w in t.split() if sum(ch.isalpha() for ch in w) >= 3)


def parse_legend(page, px_per_ft: float, box=None) -> list[Swatch]:
    x0, y0, x1, y1 = box or page.boxes["lithology_legend"]
    rgb = page.crop((x0, y0, x1, y1))
    scale = max(1.0, 300.0 / page.dpi) if page.dpi else 2.0
    out = []
    close_px = max(3, int(round((page.dpi or 150) / 100)) + 1)
    # Closing bridges broken outlines on scans, but also glues swatches stacked with a 1-2 px gap
    # into one tall blob; detect both ways and keep every rectangle found.
    nw = _nonwhite(rgb)
    rects = list(dict.fromkeys(_rects(nw, close_px) + _rects(nw, 1)))
    # Swatch-sized boxes: at least ~0.15 in on a side (letter outlines are smaller), small next to
    # the legend, and with no other swatch-sized box inside (that would be a frame).
    # Long thin bars (e.g. 0.6 x 0.08 in) count too: a letter outline is never that elongated.
    dpi = page.dpi or 150

    def swatch_sized(r) -> bool:
        lo, hi = sorted((r[2] - r[0], r[3] - r[1]))
        return (lo >= 0.15 * dpi or (lo >= 0.06 * dpi and hi >= 0.4 * dpi)) and (r[2] - r[0]) * (r[3] - r[1]) < 0.05 * nw.size

    sized = [r for r in rects if swatch_sized(r)]
    small = [r for r in sized
             if not any(q != r and r[0] <= q[0] and q[2] <= r[2] and r[1] <= q[1] and q[3] <= r[3] for q in sized)]
    # Closing can fuse a word of label text into a solid bar the size of a thin swatch: a "swatch"
    # mostly covered by a readable rock name, with paper showing between the letters, is that word.
    # (Colour can't tell: sub-pixel rendered text has coloured fringes.)
    white = classify_pixels(rgb)[1]
    words0 = [(t, bx) for t, bx in _text_lines(rgb, scale) if _is_rock_word(t)]

    def is_text(r) -> bool:
        area = (r[2] - r[0]) * (r[3] - r[1])
        for _, (a, b, c, d) in words0:
            ix = max(0.0, min(r[2], c) - max(r[0], a)) * max(0.0, min(r[3], d) - max(r[1], b))
            if ix > 0.5 * area and white[r[1] : r[3], r[0] : r[2]].mean() > 0.2:  # letters, not a filled bar
                return True
        return False

    small = [r for r in small if not is_text(r)]
    rects = [r for r in rects if not (swatch_sized(r) and is_text(r))]
    lines = _text_lines(rgb, scale, small)
    # Fill holes in the legend grid (a swatch whose outline is too broken to trace): if this adds
    # boxes, read the text again with them blanked so their pattern is not glued to a label.
    extra = [r for r in _grid_holes(small, nw)
             if not any(r[0] <= (a + c) / 2 <= r[2] and r[1] <= (b + d) / 2 <= r[3] and sum(ch.isalpha() for ch in t) >= 4
                        for t, (a, b, c, d) in lines)]  # a label written there (e.g. "COAL" beside its icon)
    if extra:
        small, rects = small + extra, rects + extra
        before, lines = lines, _text_lines(rgb, scale, small)
        # OCR is not stable under small changes to the page: keep a label it read before, outside
        # the boxes just blanked, if this pass lost it.
        def overlaps(p, q) -> bool:
            return p[0] < q[2] and q[0] < p[2] and p[1] < q[3] and q[1] < p[3]
        lines += [(t, bx) for t, bx in before if sum(ch.isalpha() for ch in t) >= 3
                  and not any(overlaps(bx, r) for r in extra) and not any(overlaps(bx, q) for _, q in lines)]
    # A rectangle holding two or more swatches (stacked bars glued by closing) is a frame, not a swatch.
    inside = lambda q, r: q != r and r[0] <= q[0] and q[2] <= r[2] and r[1] <= q[1] and q[3] <= r[3]  # noqa: E731
    rects = [r for r in rects if sum(inside(q, r) for q in small) < 2]
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
            if below_layout:
                strip = rgb[d : int(d + 0.9 * hs), a : c]
            else:  # up to the next swatch on the same row, so the next column's label is not read too
                right_edge = min([q[0] for q in small if q[0] > c and q[1] < d and q[3] > b] + [int(c + 4 * ws)]
                                 + [int(la) for t, (la, lb, lc, ld) in lines if la > c + 0.5 * ws and lb < d and ld > b])
                strip = rgb[b:d, c : right_edge - 2]
            if strip.size:
                t = _ocr_strip(strip, scale)
                if t:
                    labelled[r] = t
    # A "box" far smaller than the other swatches is a mark inside a borderless symbol (one breccia
    # clast's outline): let the borderless search below take the whole symbol instead.
    if len(labelled) >= 3:
        med = float(np.median([(r[2] - r[0]) * (r[3] - r[1]) for r in labelled]))
        labelled = {r: t for r, t in labelled.items() if (r[2] - r[0]) * (r[3] - r[1]) >= 0.25 * med}
    labelled = {r: t for r, t in labelled.items() if sum(ch.isalpha() for ch in t) >= 3}
    labelled.update(_borderless(nw, lines, labelled))
    for r, name in labelled.items():
        name = canonical_name(_clean(_untitle(name)))
        a, b, c, d = r
        inset = max(3, int(0.08 * min(c - a, d - b)))
        inner = rgb[b + inset : d - inset, a + inset : c - inset]
        if inner.size == 0:
            continue
        st, _ = row_stats(inner)
        desc = describe(st.sum(0), len(st), inner.shape[1], px_per_ft)
        is_white = bool(desc[0] < 0.01 and desc[2:-2].sum() < 0.02)
        out.append(Swatch(name, (x0 + a, y0 + b, x0 + c, y0 + d), desc, shape_descriptor(inner), is_white,
                          classify_pixels(inner)[0]))
    return out


def ink_similarity(a: np.ndarray, b: np.ndarray, tol_px: int = 2) -> float:
    """Share of each swatch's ink that the other has within tol_px (the worse of the two directions).

    1.0 means the two legend symbols are drawn identically (e.g. Clay and Shale on some logs).
    """
    h, w = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
    if h < 4 or w < 4:
        return 0.0
    A = cv2.resize(a.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    B = cv2.resize(b.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    k = np.ones((2 * tol_px + 1, 2 * tol_px + 1), np.uint8)
    dA, dB = cv2.dilate(A, k) > 0, cv2.dilate(B, k) > 0
    A, B = A > 0, B > 0
    if A.sum() < 20 or B.sum() < 20:
        return 0.0
    return float(min((A & dB).sum() / A.sum(), (B & dA).sum() / B.sum()))
