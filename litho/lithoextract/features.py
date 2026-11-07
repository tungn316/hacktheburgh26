"""Descriptors of a patterned column, comparable across scan resolutions.

Per row we keep *sufficient statistics* (pixel counts, summed stroke lengths),
so any window of rows can be described exactly by summing rows and then
deriving: ink fraction, white fraction, hue histogram of coloured fill, and the
log of mean horizontal / vertical ink stroke length in feet. Working in feet
makes a 400 DPI and a 150 DPI scan of the same symbol comparable; logs make
"dots vs dashes vs bars" differences large and scale differences small.
"""
from __future__ import annotations

import numpy as np

N_HUE = 12
# stats columns: ink, white, hue0..hue11 (coloured px), hr_sum, vr_sum
N_STATS = 2 + N_HUE + 2
FEATURE_NAMES = ["ink", "white"] + [f"hue{i}" for i in range(N_HUE)] + ["log_hrun", "log_vrun"]
TEXTURE_SCALE = 0.3  # weight of one log-unit of stroke length vs a fraction of fill
EPS_FT = 0.02


def classify_pixels(rgb: np.ndarray):
    rgb = rgb.astype(np.int16)
    mx, mn = rgb.max(2), rgb.min(2)
    ink = mx < 110
    white = (mn > 200) & (mx - mn < 30)
    colour = ~ink & ~white & (mx - mn >= 40)
    return ink, white, colour


def hue_bins(rgb: np.ndarray) -> np.ndarray:
    r, g, b = [rgb[..., i].astype(np.float32) / 255 for i in range(3)]
    mx, mn = np.maximum(np.maximum(r, g), b), np.minimum(np.minimum(r, g), b)
    d = np.where(mx - mn == 0, 1e-6, mx - mn)
    h = np.where(mx == r, ((g - b) / d) % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) / 6.0
    return np.minimum((h * N_HUE).astype(int), N_HUE - 1)


def run_length_map(mask: np.ndarray, axis: int) -> np.ndarray:
    """For each True pixel, the length of the run of True pixels it belongs to along ``axis``."""
    m = mask if axis == 1 else mask.T
    padded = np.concatenate([m, np.zeros((m.shape[0], 1), bool)], axis=1).ravel()
    edges = np.diff(np.concatenate([[0], padded.astype(np.int8), [0]]))
    starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
    lengths = ends - starts
    out = np.zeros(padded.shape, np.float32)
    out[padded] = np.repeat(lengths, lengths)
    out = out.reshape(m.shape[0], m.shape[1] + 1)[:, :-1]
    return out if axis == 1 else out.T


def interior(rgb: np.ndarray, margin: int = 2) -> np.ndarray:
    """Strip the column's own vertical border lines (columns that are ink most of the way down)."""
    ink, _, _ = classify_pixels(rgb)
    border = ink.mean(0) > 0.6
    cols = np.where(~border)[0]
    if len(cols) == 0:
        return rgb
    run = max(np.split(cols, np.where(np.diff(cols) != 1)[0] + 1), key=len)
    a, b = run[0] + margin, run[-1] + 1 - margin
    return rgb[:, a:b] if b > a else rgb[:, run[0] : run[-1] + 1]


def row_stats(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-row sufficient statistics, with horizontal rule rows (grid lines, box edges) patched over.

    Returns (stats[n_rows, N_STATS], is_rule[n_rows]).
    """
    ink, white, colour = classify_pixels(rgb)
    w = rgb.shape[1]
    st = np.zeros((rgb.shape[0], N_STATS), np.float64)
    st[:, 0] = ink.sum(1)
    st[:, 1] = white.sum(1)
    hb = hue_bins(rgb)
    for k in range(N_HUE):
        st[:, 2 + k] = ((hb == k) & colour).sum(1)
    hr = run_length_map(ink, 1)
    st[:, -2] = hr.sum(1)
    st[:, -1] = run_length_map(ink, 0).sum(1)
    # A row that is mostly one long horizontal stroke is a ruled line, not pattern.
    is_rule = hr.max(1) >= 0.7 * w
    # Patch rule rows (and their 1-px neighbours: anti-aliasing) with the nearest clean row above/below.
    bad = is_rule | np.roll(is_rule, 1) | np.roll(is_rule, -1)
    good_idx = np.where(~bad)[0]
    if len(good_idx):
        bad_idx = np.where(bad)[0]
        pos = np.searchsorted(good_idx, bad_idx)
        above = good_idx[np.clip(pos - 1, 0, len(good_idx) - 1)]
        below = good_idx[np.clip(pos, 0, len(good_idx) - 1)]
        nearest = np.where(np.abs(bad_idx - above) <= np.abs(below - bad_idx), above, below)
        st[bad_idx] = st[nearest]
        # vertical runs through a rule are inflated; recompute is overkill, cap instead
    return st, is_rule


def describe(stats_sum: np.ndarray, n_rows: np.ndarray | float, width: int, px_per_ft: float) -> np.ndarray:
    """Turn summed stats (…, N_STATS) over n_rows rows into feature vectors (…, n_features)."""
    stats_sum = np.asarray(stats_sum, np.float64)
    area = np.maximum(np.asarray(n_rows, np.float64) * width, 1.0)[..., None]
    fr = stats_sum[..., : 2 + N_HUE] / area
    ink_px = np.maximum(stats_sum[..., 0:1], 1.0)
    runs_ft = stats_sum[..., -2:] / ink_px / px_per_ft
    tex = np.log(runs_ft + EPS_FT)
    # With almost no ink, stroke length is meaningless: pin it to the "no strokes" value.
    tex = np.where(fr[..., 0:1] < 0.005, np.log(EPS_FT), tex)
    return np.concatenate([fr, TEXTURE_SCALE * tex], axis=-1)
