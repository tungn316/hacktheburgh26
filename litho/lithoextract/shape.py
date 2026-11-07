"""Scale-free pattern descriptor for matching column intervals to legend swatches.

The legend draws a symbol at a different size and spacing from the column, so we
describe *what kind of strokes* the pattern is made of, not how big they are:
for each ink pixel, is it part of a horizontal stroke, a vertical stroke, a
diagonal stroke, or a dot? Plus fill colour and how solid the fill is.
"""
from __future__ import annotations

import numpy as np

from .features import N_HUE, classify_pixels, hue_bins, run_length_map

SHAPE_NAMES = ["horiz", "vert", "diag", "anti", "dot", "solid", "empty"] + [f"hue{i}" for i in range(N_HUE)] + ["ink"]


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
    d[-1] = ink_frac
    return d


def shape_distance(a: np.ndarray, b: np.ndarray) -> float:
    w = np.ones(len(a), np.float32)
    w[7:-1] = 1.5  # colour is the most reliable cue when present
    w[-1] = 2.0  # ink density: separates dot-density variants (fine vs medium sand)
    return float(np.abs((a - b) * w).sum())
