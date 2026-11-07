"""Depth calibration: read the depth labels with OCR and fit pixel row -> depth.

Works for horizontal labels and for labels rotated 90 degrees either way: we try
each orientation and keep the one whose numbers form the best straight line.
Only black ink is OCR'd, so secondary coloured labels (e.g. blue TVD values
printed under MD values) are ignored.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pytesseract
from PIL import Image

from .io import Page, ink_mask


@dataclass
class DepthFit:
    slope: float  # ft per pixel row
    intercept: float  # depth at absolute row 0
    points: list[tuple[float, float]]  # (absolute y, depth) inliers
    n_read: int
    rotation: int
    residual_ft: float

    def depth(self, y: float) -> float:
        return self.intercept + self.slope * y

    @property
    def px_per_ft(self) -> float:
        return 1.0 / self.slope


def _ocr_numbers(img: np.ndarray, scale: float) -> list[tuple[float, float, float]]:
    """Return (center_x, center_y, value) for each number tesseract finds in a binary image."""
    pil = Image.fromarray(np.where(img, 0, 255).astype(np.uint8))
    if scale != 1:
        pil = pil.resize((int(pil.width * scale), int(pil.height * scale)), Image.LANCZOS)
    data = pytesseract.image_to_data(
        pil,
        config="--psm 11 -c tessedit_char_whitelist=0123456789.",
        output_type=pytesseract.Output.DICT,
    )
    out = []
    for txt, conf, l, t, w, h in zip(data["text"], data["conf"], data["left"], data["top"], data["width"], data["height"]):
        m = re.fullmatch(r"(\d{3,6})(\.0)?", txt.strip())
        if m and float(conf) > 30:
            out.append(((l + w / 2) / scale, (t + h / 2) / scale, float(m.group(1))))
    return out


def _robust_line(ys: np.ndarray, ds: np.ndarray, tol_ft: float) -> tuple[float, float, np.ndarray]:
    """RANSAC-style line fit over all point pairs; returns slope, intercept, inlier mask."""
    best = (0.0, 0.0, np.zeros(len(ys), bool))
    n = len(ys)
    for i in range(n):
        for j in range(i + 1, n):
            if ys[j] == ys[i]:
                continue
            s = (ds[j] - ds[i]) / (ys[j] - ys[i])
            if s <= 0:
                continue
            c = ds[i] - s * ys[i]
            inl = np.abs(ds - (s * ys + c)) < tol_ft
            if inl.sum() > best[2].sum():
                best = (s, c, inl)
    s, c, inl = best
    if inl.sum() >= 2:
        s, c = np.polyfit(ys[inl], ds[inl], 1)
        inl = np.abs(ds - (s * ys + c)) < tol_ft
    return float(s), float(c), inl


def calibrate_depth(page: Page, box: tuple[int, int, int, int] | None = None, chunk: int = 4000) -> DepthFit:
    x0, y0, x1, y1 = box or page.boxes["depth_column"]
    pad = 40  # labels can sit slightly outside the box vertically
    y0p, y1p = max(0, y0 - pad), min(page.image.height, y1 + pad)
    ink = ink_mask(page.crop((x0, y0p, x1, y1p)))
    # Small text at low DPI needs upscaling for tesseract.
    scale = max(1.0, 300.0 / page.dpi) if page.dpi else 2.0

    best: DepthFit | None = None
    for rot in (0, 90, 270):
        pts = []
        for c0 in range(0, ink.shape[0], chunk):
            part = ink[c0 : c0 + chunk + 200]
            if rot:
                part = np.rot90(part, k=rot // 90)  # rot90 is counter-clockwise
            for cx, cy, val in _ocr_numbers(part, scale):
                # map back to the unrotated chunk's row coordinate
                # np.rot90(k=1) maps (row, col) -> (W-1-col, row): original row = new x.
                # np.rot90(k=3) maps (row, col) -> (col, H-1-row): original row = H-1-new x.
                if rot == 0:
                    yy = cy
                elif rot == 90:
                    yy = cx
                else:
                    yy = part.shape[1] - 1 - cx
                pts.append((y0p + c0 + yy, val))
        if len(pts) < 2:
            continue
        pts = sorted(set((round(y, 1), v) for y, v in pts))
        ys = np.array([p[0] for p in pts], float)
        ds = np.array([p[1] for p in pts], float)
        expected = (page.dpi * 5 / 100) if page.dpi else None  # "5 inch per 100 ft" logs
        s, c, inl = _robust_line(ys, ds, tol_ft=2.0)
        if inl.sum() < 2:
            continue
        resid = float(np.abs(ds[inl] - (s * ys[inl] + c)).mean())
        fit = DepthFit(s, c, [(float(y), float(d)) for y, d in zip(ys[inl], ds[inl])], len(pts), rot, resid)
        if best is None or len(fit.points) > len(best.points):
            best = fit
    if best is None:
        raise RuntimeError("could not read enough depth labels")
    return best
