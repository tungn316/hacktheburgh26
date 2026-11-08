"""Scale-free pattern descriptor for matching column intervals to legend swatches.

The legend draws a symbol at a different size and spacing from the column, so we
describe *what kind of strokes* the pattern is made of, not how big they are:
for each ink pixel, is it part of a horizontal stroke, a vertical stroke, a
diagonal stroke, or a dot? Plus fill colour and how solid the fill is.
"""
from __future__ import annotations

import numpy as np

from .features import N_HUE, classify_pixels, hue_bins, run_length_map

SHAPE_NAMES = ["horiz", "vert", "diag", "anti", "dot", "solid", "empty"] + [f"hue{i}" for i in range(N_HUE)] + ["fill", "ink"]
FILL = len(SHAPE_NAMES) - 2


def _diag_runs(mask: np.ndarray, anti: bool) -> np.ndarray:
    h, w = mask.shape
    i, j = np.nonzero(mask)
    col = (j + i) if anti else (j - i + h)
    S = np.zeros((h, h + w + 1), bool)
    S[i, col] = True
    R = run_length_map(S, 0)
    out = np.zeros(mask.shape, np.float32)
    out[i, j] = R[i, col]
    return out


def shape_descriptor(rgb: np.ndarray) -> np.ndarray:
    ink, white, colour = classify_pixels(rgb)
    # Rows inked across (nearly) the whole width are where a symbol's tiles meet, or ruled lines:
    # the column repeats a symbol's edge every tile, the legend's cropped swatch has none.
    # Only when such rows are sparse (one per tile, < 20% of rows) and on colour fills: a symbol can
    # itself be full-width lines (shale drawn as dense lines; black-and-white shale).
    rule = ink.mean(axis=1) >= 0.8
    if 0 < rule.mean() < 0.2 and colour[~rule].sum() > 0.5 * max(1, (~ink[~rule]).sum()):
        rgb, ink, white, colour = rgb[~rule], ink[~rule], white[~rule], colour[~rule]
    n_ink = ink.sum()
    d = np.zeros(len(SHAPE_NAMES), np.float32)
    area = ink.size
    ink_frac = n_ink / max(area, 1)
    d[5] = min(1.0, max(0.0, (ink_frac - 0.5) / 0.4))  # solid fill (coal)
    d[6] = 1.0 if ink_frac < 0.01 and colour.mean() < 0.02 else 0.0  # empty / white
    if n_ink >= 20 and d[5] < 1:
        hr, vr = run_length_map(ink, 1)[ink], run_length_map(ink, 0)[ink]
        dr, ar = _diag_runs(ink, False)[ink], _diag_runs(ink, True)[ink]
        t = max(1.0, float(np.median(np.minimum(hr, vr))))  # stroke thickness
        longest = np.maximum.reduce([hr, vr, dr, ar])
        cls_h = (hr == longest) & (hr >= 3 * t)
        cls_v = (vr == longest) & (vr >= 3 * t) & ~cls_h
        cls_d = (dr == longest) & (dr >= 3 * t) & ~cls_h & ~cls_v
        cls_a = (ar == longest) & (ar >= 3 * t) & ~cls_h & ~cls_v & ~cls_d
        cls_dot = longest < 3 * t
        total = len(hr)
        d[0:5] = [c.sum() / total for c in (cls_h, cls_v, cls_d, cls_a, cls_dot)]
        d[0:5] *= 1 - d[5]
    if colour.any():
        hb = hue_bins(rgb)
        nonink = max(1, (~ink).sum())
        for k in range(N_HUE):
            d[7 + k] = ((hb == k) & colour).sum() / nonink
    # How dark the paper/fill is (white ~0, light grey ~0.4, a strong orange ~1.1): tells white from
    # grey shale fills and a bright orange from a pale peach of similar hue.
    if (~ink).any():
        med = np.median(rgb[~ink].reshape(-1, 3), axis=0).astype(np.float32)
        d[FILL] = min(1.5, max(0.0, (255 - float(med @ np.array([0.3, 0.59, 0.11], np.float32))) / 100))
    d[-1] = ink_frac
    return d


def shape_distance(a: np.ndarray, b: np.ndarray) -> float:
    w = np.ones(len(a), np.float32)
    w[7:FILL] = 1.5  # colour is the most reliable cue when present
    w[-1] = 2.0  # ink density: separates dot-density variants (fine vs medium sand)
    return float(np.abs((a - b) * w).sum())
