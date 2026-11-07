"""Snap interval boundaries to the log's depth grid when the log was drawn on one.

Many logs only change lithology on round depths (every 10 ft on a 5 in/100 ft
log). If most boundaries sit on a regular step we take that as the grid and
move boundaries that are near a grid line onto it. Pages without such a step
are left alone.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import comb

import numpy as np

STEPS = (100, 50, 20, 10, 5, 2, 1)  # candidate grid steps, ft
TOL_FRAC = 0.15  # a boundary within 15% of a step from a grid line counts as on the grid
MIN_FRAC = 0.6  # "most" boundaries must be on the grid (chance alone gives ~2 * TOL_FRAC = 30%)
MIN_BOUNDARIES = 4
MIN_TOL_PX = 3.0  # a grid finer than the boundary-placement noise can't be detected
MAX_P = 1e-3  # and that many on-grid boundaries must be very unlikely by chance


@dataclass
class Grid:
    step: float
    tol: float
    on_grid_frac: float

    def nearest(self, d: float) -> float:
        return round(d / self.step) * self.step

    def on(self, d: float) -> bool:
        return abs(d - self.nearest(d)) <= self.tol


def find_grid(depths, px_per_ft: float) -> Grid | None:
    d = np.asarray(depths, float)
    if len(d) < MIN_BOUNDARIES:
        return None
    best: Grid | None = None
    for step in STEPS:  # largest first, so a near-tie goes to the coarser grid
        tol = TOL_FRAC * step
        if tol * px_per_ft < MIN_TOL_PX:
            continue
        hits = int((np.abs(d - np.round(d / step) * step) <= tol).sum())
        frac = hits / len(d)
        if frac >= MIN_FRAC and _p_chance(hits, len(d), 2 * TOL_FRAC) < MAX_P and (best is None or frac > best.on_grid_frac + 0.05):
            best = Grid(float(step), tol, frac)
    return best


def _p_chance(k: int, n: int, p: float) -> float:
    """P(at least k of n random boundaries land on the grid)."""
    return sum(comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k, n + 1))
