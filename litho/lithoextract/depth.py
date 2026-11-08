"""Depth calibration: read the depth labels with OCR and map pixel row -> depth.

* Only black ink is OCR'd, so secondary coloured labels (e.g. blue TVD values
  printed beside MD values) are ignored.
* The label orientation (0 / 90 / 270 degrees) is decided once, on a couple of
  probe chunks, and then used for the whole column.
* Very tall columns are only *sampled*: an evenly spaced subset of chunks
  (always including the first and last) is OCR'd. The mapping only needs a few
  dozen good labels, and OCR is most of the run time.
  If that sample yields too few good labels, the remaining chunks are read too.
* Bad readings are thrown out: labels must lie on the dominant straight line
  (RANSAC over plausible scales for the scan's DPI, 2 ft tolerance) and increase
  monotonically. This also ignores a second, parallel label series.
* Depth is then interpolated piecewise-linearly between neighbouring labels
  (extrapolated with the overall scale beyond the first / last label), so small
  drift in the scan is followed rather than averaged.
* If a stretch of the rejected labels follows a different scale where the main
  line has no labels, the depth scale changes part-way down: that is reported as
  a warning, as is a fit resting on fewer than 4 labels.
"""
from __future__ import annotations

import bisect
import re
from dataclasses import dataclass, field

import numpy as np
import pytesseract
from PIL import Image

from .io import Page, ink_mask

LOCAL_TOL_FT = 2.0  # a label may sit this far from the fitted line
SCALE_CHANGE = 0.08  # warn when the local scale differs from the overall one by more than this


@dataclass
class DepthFit:
    ys: np.ndarray  # absolute pixel rows of the accepted labels (increasing)
    ds: np.ndarray  # their depths (strictly increasing)
    slope: float  # overall ft per pixel row (robust fit), used to extrapolate
    intercept: float
    n_read: int
    n_rejected: int
    rotation: int
    residual_ft: float  # mean |label - interpolated line through its neighbours|
    chunks_read: int
    chunks_total: int
    warnings: list[str] = field(default_factory=list)

    @property
    def px_per_ft(self) -> float:
        return 1.0 / self.slope

    @property
    def points(self) -> list[tuple[float, float]]:
        return list(zip(self.ys.tolist(), self.ds.tolist()))

    def depth(self, y):
        y = np.asarray(y, float)
        d = np.interp(y, self.ys, self.ds)
        d = np.where(y < self.ys[0], self.ds[0] + self.slope * (y - self.ys[0]), d)
        d = np.where(y > self.ys[-1], self.ds[-1] + self.slope * (y - self.ys[-1]), d)
        return float(d) if d.ndim == 0 else d

    def row(self, depth):
        """Inverse of depth(): the (fractional) pixel row of a depth."""
        d = np.asarray(depth, float)
        y = np.interp(d, self.ds, self.ys)
        y = np.where(d < self.ds[0], self.ys[0] + (d - self.ds[0]) / self.slope, y)
        y = np.where(d > self.ds[-1], self.ys[-1] + (d - self.ds[-1]) / self.slope, y)
        return float(y) if y.ndim == 0 else y


def _ocr_numbers(img: np.ndarray, scale: float) -> list[tuple[float, float, float]]:
    """Return (center_x, center_y, value) for each number tesseract finds in a binary image."""
    pil = Image.fromarray(np.where(img, 0, 255).astype(np.uint8))
    if scale != 1:
        pil = pil.resize((int(pil.width * scale), int(pil.height * scale)), Image.LANCZOS)
    data = pytesseract.image_to_data(
        pil,
        config="--psm 11 -c tessedit_char_whitelist=0123456789.,",
        output_type=pytesseract.Output.DICT,
    )
    out = []
    for txt, conf, l, t, w, h in zip(data["text"], data["conf"], data["left"], data["top"], data["width"], data["height"]):
        # Plain "9120" or with thousands separators "9,120" (a stray trailing comma/dot is dropped).
        m = re.fullmatch(r"(\d{3,6}|\d{1,3}(?:,\d{3})+)(\.0)?", txt.strip().rstrip(",."))
        if m and float(conf) > 30:
            out.append(((l + w / 2) / scale, (t + h / 2) / scale, float(m.group(1).replace(",", ""))))
    return out


def _read_chunk(ink: np.ndarray, rot: int, scale: float) -> list[tuple[float, float]]:
    """(row within the chunk, value) for every number read with the text rotated by ``rot``."""
    part = np.rot90(ink, k=rot // 90) if rot else ink  # rot90 is counter-clockwise
    out = []
    for cx, cy, val in _ocr_numbers(part, scale):
        # np.rot90(k=1) maps (row, col) -> (W-1-col, row): original row = new x.
        # np.rot90(k=3) maps (row, col) -> (col, H-1-row): original row = H-1-new x.
        yy = cy if rot == 0 else cx if rot == 90 else part.shape[1] - 1 - cx
        out.append((yy, val))
    return out


def _dedupe(pts: list[tuple[float, float]], near_px: float) -> tuple[np.ndarray, np.ndarray]:
    """Sort by row; the same value read twice (overlapping chunks) within near_px becomes one point."""
    pts = sorted(pts)
    ys: list[float] = []
    ds: list[float] = []
    for y, v in pts:
        dup = None
        for k in range(len(ys) - 1, -1, -1):
            if y - ys[k] >= near_px:
                break
            if ds[k] == v:
                dup = k
                break
        if dup is None:
            ys.append(y)
            ds.append(v)
        else:
            ys[dup] = (ys[dup] + y) / 2
    return np.array(ys, float), np.array(ds, float)


def _slope_range(dpi: float) -> tuple[float, float]:
    """Plausible ft-per-pixel range: from ~20 in per 100 ft down to ~0.2 in per 100 ft."""
    lo_ppf, hi_ppf = (0.002 * dpi, 0.25 * dpi) if dpi else (0.1, 200.0)
    return 1.0 / hi_ppf, 1.0 / lo_ppf


def _ransac(ys: np.ndarray, ds: np.ndarray, tol: float, srange: tuple[float, float],
            n_pairs: int = 4000, seed: int = 0):
    """Line through the most labels (within tol ft). Returns (slope, intercept, inlier mask) or None."""
    n = len(ys)
    if n < 2:
        return None
    i, j = np.triu_indices(n, 1)
    if len(i) > n_pairs:
        sel = np.random.default_rng(seed).choice(len(i), n_pairs, replace=False)
        i, j = i[sel], j[sel]
    dy = ys[j] - ys[i]
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(dy != 0, (ds[j] - ds[i]) / dy, 0.0)
    ok = (s >= srange[0]) & (s <= srange[1])
    if not ok.any():
        return None
    s, c = s[ok], ds[i[ok]] - s[ok] * ys[i[ok]]
    best, bs, bc = -1, 0.0, 0.0
    for b in range(0, len(s), 512):
        cnt = (np.abs(ds[None, :] - (s[b : b + 512, None] * ys[None, :] + c[b : b + 512, None])) < tol).sum(1)
        k = int(np.argmax(cnt))
        if cnt[k] > best:
            best, bs, bc = int(cnt[k]), float(s[b + k]), float(c[b + k])
    inl = np.abs(ds - (bs * ys + bc)) < tol
    for _ in range(2):  # refine on the inliers
        if inl.sum() >= 3:
            fs, fc = np.polyfit(ys[inl], ds[inl], 1)
            new = np.abs(ds - (fs * ys + fc)) < tol
            if new.sum() >= inl.sum():
                bs, bc, inl = float(fs), float(fc), new
    return bs, bc, inl


def _filter(ys: np.ndarray, ds: np.ndarray, srange: tuple[float, float]):
    """Accepted labels: on the dominant straight line, and increasing in depth down the page.

    Returns (slope, intercept, mask) or None.
    """
    fit = _ransac(ys, ds, LOCAL_TOL_FT, srange)
    if fit is None:
        return None
    s, c, inl = fit
    idx = np.where(inl)[0]
    keep = np.zeros(len(ys), bool)
    if len(idx):
        keep[idx[_longest_increasing(ds[idx])]] = True
    return s, c, keep


def _longest_increasing(v: np.ndarray) -> list[int]:
    """Indices of a longest strictly increasing subsequence of v (patience sorting, O(n log n))."""
    tails: list[float] = []
    tail_idx: list[int] = []
    prev = [-1] * len(v)
    for i, x in enumerate(v):
        k = bisect.bisect_left(tails, x)
        if k == len(tails):
            tails.append(x)
            tail_idx.append(i)
        else:
            tails[k], tail_idx[k] = x, i
        prev[i] = tail_idx[k - 1] if k else -1
    out, i = [], tail_idx[-1] if tail_idx else -1
    while i >= 0:
        out.append(i)
        i = prev[i]
    return out[::-1]


def _scale_warnings(ys: np.ndarray, ds: np.ndarray, keep: np.ndarray, slope: float,
                    srange: tuple[float, float], win: int = 8) -> list[str]:
    """Report stretches where the labels follow a different scale from the main line.

    Rejected labels are scanned in runs of ``win``; a run that is itself a clean line with a
    different scale, in a part of the column where the main line has (almost) no labels, means
    the depth scale changes there. (A second label series printed alongside the main one,
    e.g. TVD next to MD, overlaps the main line's labels and is not reported.)
    """
    out: list[str] = []
    rej = np.where(~keep)[0]
    last_end = -np.inf
    for a in range(0, max(0, len(rej) - win + 1), max(1, win // 2)):
        r = rej[a : a + win]
        y, d = ys[r], ds[r]
        if y[0] <= last_end:
            continue
        fit = _ransac(y, d, LOCAL_TOL_FT, srange)
        if fit is None or fit[2].sum() < 0.75 * win:
            continue
        s = fit[0]
        in_span = (ys >= y[0]) & (ys <= y[-1])
        if abs(s / slope - 1) > SCALE_CHANGE and keep[in_span].mean() < 0.3:
            out.append(f"depth scale changes near {d[fit[2]][0]:.0f} ft: labels there follow "
                       f"{1 / s:.3f} px/ft vs {1 / slope:.3f} px/ft for the rest; depths there are extrapolated")
            last_end = y[-1]
    return out


def calibrate_depth(
    page: Page,
    box: tuple[int, int, int, int] | None = None,
    *,
    chunk: int = 4000,
    overlap: int = 200,
    max_chunks: int = 12,
    n_probe: int = 2,
    min_labels: int = 30,
) -> DepthFit:
    x0, y0, x1, y1 = box or page.boxes["depth_column"]
    pad = 40  # labels can sit slightly outside the box vertically
    y0p, y1p = max(0, y0 - pad), min(page.height, y1 + pad)
    scale = max(1.0, 300.0 / page.dpi) if page.dpi else 2.0  # small text at low DPI needs upscaling
    srange = _slope_range(page.dpi)

    starts = list(range(y0p, y1p, chunk))
    if len(starts) > max_chunks:  # very tall column: read an evenly spaced sample of chunks
        pick = np.unique(np.round(np.linspace(0, len(starts) - 1, max_chunks)).astype(int))
        sampled = [starts[i] for i in pick]
    else:
        sampled = list(starts)

    readings: dict[tuple[int, int], list[tuple[float, float]]] = {}

    def read(c0s, rot):
        pts = []
        for c0 in c0s:
            if (c0, rot) not in readings:
                ink = ink_mask(page.crop((x0, c0, x1, min(y1p, c0 + chunk + overlap))))
                readings[(c0, rot)] = [(c0 + yy, v) for yy, v in _read_chunk(ink, rot, scale)]
            pts += readings[(c0, rot)]
        return pts

    def score(pts) -> int:
        ys, ds = _dedupe(pts, overlap)
        f = _filter(ys, ds, srange)
        return int(f[2].sum()) if f else 0

    # Decide the label orientation once, on probe chunks from the middle of the sample;
    # if they are inconclusive, judge each orientation on the whole sample instead.
    probe = sampled if len(sampled) <= n_probe else \
        [sampled[i] for i in np.linspace(0, len(sampled) - 1, n_probe + 2).round().astype(int)[1:-1]]
    scores = {rot: score(read(probe, rot)) for rot in (0, 90, 270)}
    ranked = sorted(scores, key=lambda r: (-scores[r], r != 0))
    if scores[ranked[0]] < 3 or scores[ranked[0]] == scores[ranked[1]]:
        scores = {rot: score(read(sampled, rot)) for rot in (0, 90, 270)}
        ranked = sorted(scores, key=lambda r: (-scores[r], r != 0))
    rot = ranked[0]

    to_read = sampled
    pts = read(to_read, rot)
    if score(pts) < min_labels and len(sampled) < len(starts):  # sparse labels: read every chunk after all
        to_read = starts
        pts = read(to_read, rot)
    ys, ds = _dedupe(pts, overlap)
    if len(ys) < 2:
        raise RuntimeError("could not read enough depth labels")
    f = _filter(ys, ds, srange)
    if f is None or f[2].sum() < 2:
        raise RuntimeError("could not read enough consistent depth labels")
    s, c, keep = f
    ky, kd = ys[keep], ds[keep]

    # How well does each label sit on the line through its two neighbours?
    if len(ky) >= 3:
        pred = kd[:-2] + (kd[2:] - kd[:-2]) * (ky[1:-1] - ky[:-2]) / (ky[2:] - ky[:-2])
        resid = float(np.abs(kd[1:-1] - pred).mean())
    else:
        resid = 0.0
    warnings = _scale_warnings(ys, ds, keep, s, srange)
    if len(ky) < 4:
        warnings.append(f"only {len(ky)} depth labels agree on a scale ({1 / s:.3f} px/ft): depths are unreliable")
    return DepthFit(ky, kd, float(s), float(c), len(ys), int((~keep).sum()), rot, resid,
                    len(to_read), len(starts), warnings)
