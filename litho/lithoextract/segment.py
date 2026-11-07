"""Split a lithology column into intervals and group intervals that share a pattern."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .features import describe, row_stats


@dataclass
class Segment:
    y0: int  # rows relative to the column crop
    y1: int
    descriptor: np.ndarray
    blank: bool = False
    group: int = -1
    meta: dict = field(default_factory=dict)


class Column:
    """A column crop with cumulative stats so any row range is described in O(1)."""

    def __init__(self, rgb: np.ndarray, px_per_ft: float) -> None:
        self.width = rgb.shape[1]
        self.ppf = px_per_ft
        self.stats, self.is_rule = row_stats(rgb)
        self.n = len(self.stats)
        self.cum = np.zeros((self.n + 1, self.stats.shape[1]))
        np.cumsum(self.stats, axis=0, out=self.cum[1:])

    def desc(self, a, b) -> np.ndarray:
        a, b = np.asarray(a), np.asarray(b)
        return describe(self.cum[b] - self.cum[a], np.maximum(b - a, 1), self.width, self.ppf)

    def change_score(self, half: int) -> np.ndarray:
        r = np.arange(self.n)
        above = self.desc(np.clip(r - half, 0, self.n), r)
        below = self.desc(r, np.clip(r + half, 0, self.n))
        s = np.linalg.norm(above - below, axis=1)
        s[: half // 2] = 0
        s[self.n - half // 2 :] = 0
        return s


def _peaks(score: np.ndarray, thr: float, min_dist: int) -> list[int]:
    taken = np.zeros(len(score), bool)
    out = []
    for i in np.argsort(-score):
        if score[i] < thr:
            break
        if taken[max(0, i - min_dist) : i + min_dist].any():
            continue
        taken[i] = True
        out.append(int(i))
    return sorted(out)


def _refine(col: Column, b: int, half: int) -> int:
    """Snap a boundary to the sharpest change nearby, using a smaller window."""
    small = max(2, half // 3)
    lo, hi = max(small, b - half // 2), min(col.n - small, b + half // 2)
    if hi <= lo:
        return b
    r = np.arange(lo, hi)
    s = np.linalg.norm(col.desc(r - small, r) - col.desc(r, r + small), axis=1)
    return int(r[np.argmax(s)])


def segment_column(
    rgb: np.ndarray,
    px_per_ft: float,
    *,
    window_ft: float = 2.0,
    thr: float = 0.22,
    merge_thr: float = 0.12,
    min_ft: float = 1.0,
    blank_ink: float = 0.01,
) -> tuple[list[Segment], Column]:
    col = Column(rgb, px_per_ft)
    half = max(4, int(round(window_ft * px_per_ft)))
    score = col.change_score(half)
    cuts = sorted(set([0, *[_refine(col, b, half) for b in _peaks(score, thr, half)], col.n]))

    segs: list[Segment] = []
    min_rows = max(2, int(min_ft * px_per_ft))
    for a, b in zip(cuts[:-1], cuts[1:]):
        m = (b - a) // 6  # describe the middle, away from blurred edges
        d = col.desc(a + m, b - m)
        blank = bool(d[0] < blank_ink and d[2:-2].sum() < 0.02)
        segs.append(Segment(a, b, d, blank))

    # Absorb slivers thinner than min_ft into the more similar neighbour, then merge look-alikes.
    changed = True
    while changed:
        changed = False
        for i, s in enumerate(segs):
            if s.y1 - s.y0 >= min_rows or len(segs) == 1:
                continue
            nb = [j for j in (i - 1, i + 1) if 0 <= j < len(segs)]
            j = min(nb, key=lambda j: np.linalg.norm(segs[j].descriptor - s.descriptor))
            t = segs[j]
            t.y0, t.y1 = min(t.y0, s.y0), max(t.y1, s.y1)
            del segs[i]
            changed = True
            break
    merged: list[Segment] = []
    for s in segs:
        if merged and merged[-1].blank == s.blank and np.linalg.norm(merged[-1].descriptor - s.descriptor) < merge_thr:
            p = merged[-1]
            p.y1 = s.y1
            m = (p.y1 - p.y0) // 6
            p.descriptor = col.desc(p.y0 + m, p.y1 - m)
        else:
            merged.append(s)
    for s in merged:
        m = (s.y1 - s.y0) // 6
        s.descriptor = col.desc(s.y0 + m, s.y1 - m)
    return merged, col


def group_segments(segs: list[Segment], thr: float = 0.15) -> int:
    """Greedy agglomerative grouping on length-weighted centroids. Returns the number of groups."""
    groups: list[tuple[np.ndarray, float]] = []
    for s in sorted([s for s in segs if not s.blank], key=lambda s: -(s.y1 - s.y0)):
        L = float(s.y1 - s.y0)
        dists = [np.linalg.norm(c - s.descriptor) for c, _ in groups]
        if dists and min(dists) < thr:
            g = int(np.argmin(dists))
            c, n = groups[g]
            groups[g] = ((c * n + s.descriptor * L) / (n + L), n + L)
        else:
            groups.append((s.descriptor.copy(), L))
            g = len(groups) - 1
        s.group = g
    return len(groups)
