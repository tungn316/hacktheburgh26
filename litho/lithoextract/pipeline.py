"""End to end: image + boxes -> depth-registered, legend-labelled lithology intervals."""
from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .depth import calibrate_depth
from .features import interior
from .io import Page, load_page
from .legend import parse_legend
from .segment import group_segments, segment_column
from .shape import shape_descriptor, shape_distance

AMBIGUOUS_MARGIN = 0.10  # best and runner-up legend matches closer than this → flag as ambiguous


@dataclass
class Interval:
    top_ft: float
    base_ft: float
    group: int
    lithology: str
    confidence: float
    alternatives: list[str]
    ambiguous: bool
    y_top_px: int
    y_base_px: int


def _group_shape(rgb: np.ndarray, segs, group: int, max_rows: int = 4000) -> np.ndarray:
    parts, used = [], 0
    for s in sorted([s for s in segs if s.group == group], key=lambda s: -(s.y1 - s.y0)):
        m = (s.y1 - s.y0) // 5
        part = rgb[s.y0 + m : s.y1 - m]
        if len(part) == 0:
            continue
        parts.append(part[: max_rows - used])
        used += len(parts[-1])
        if used >= max_rows:
            break
    return shape_descriptor(np.concatenate(parts)) if parts else np.zeros(1)


def run_page(page: Page) -> dict:
    t0 = time.time()
    fit = calibrate_depth(page)
    ppf = fit.px_per_ft

    x0, y0, x1, y1 = page.boxes["interpreted_lithology_column"]
    rgb = interior(page.crop((x0, y0, x1, y1)))
    segs, _ = segment_column(rgb, ppf)
    n_groups = group_segments(segs)

    swatches = parse_legend(page, ppf)
    white = [s for s in swatches if s.is_white]

    # Map each group to its nearest legend swatch by scale-free stroke shape + colour.
    group_label: dict[int, tuple[str, float, list[str], bool]] = {}
    for g in range(n_groups):
        gd = _group_shape(rgb, segs, g)
        ranked = sorted(((shape_distance(gd, s.shape), s.name) for s in swatches if not s.is_white))
        if not ranked:
            group_label[g] = (f"group_{g}", 0.0, [], True)
            continue
        d1, n1 = ranked[0]
        d2 = ranked[1][0] if len(ranked) > 1 else d1 + 1.0
        margin = d2 - d1
        conf = float(np.clip(margin / (d1 + margin + 1e-6), 0, 1))
        alts = [n for d, n in ranked[1:4] if d - d1 < 0.5]
        group_label[g] = (n1, round(conf, 3), alts, margin < AMBIGUOUS_MARGIN)

    intervals: list[Interval] = []
    for i, s in enumerate(segs):
        top, base = fit.depth(y0 + s.y0), fit.depth(y0 + s.y1)
        if s.blank:
            at_end = i == 0 or i == len(segs) - 1
            if at_end:
                continue  # outside the logged range: no data
            if white:
                name, conf, alts, amb = white[0].name, 0.8, [], False
            else:
                name, conf, alts, amb = "Blank (no pattern)", 0.5, [], True
            g = -1
        else:
            g = s.group
            name, conf, alts, amb = group_label[g]
        intervals.append(Interval(round(top, 2), round(base, 2), g, name, conf, alts, amb, y0 + s.y0, y0 + s.y1))

    return {
        "image": page.path.name,
        "depth_fit": {
            "px_per_ft": round(ppf, 4),
            "labels_used": len(fit.points),
            "labels_read": fit.n_read,
            "mean_residual_ft": round(fit.residual_ft, 3),
            "label_rotation_deg": fit.rotation,
        },
        "legend": [{"name": s.name, "box": list(s.box), "white": s.is_white} for s in swatches],
        "n_groups": n_groups,
        "intervals": [asdict(iv) for iv in intervals],
        "seconds": round(time.time() - t0, 1),
    }


PALETTE = [(230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180), (70, 240, 240),
           (240, 50, 230), (210, 245, 60), (128, 128, 0), (0, 128, 128)]


def render_overlay(page: Page, result: dict, out: Path, top_ft: float | None = None, span_ft: float = 400) -> None:
    """Scan of depth + lithology columns beside the extracted intervals (coloured by group)."""
    dx0 = page.boxes["depth_column"][0]
    lx1 = page.boxes["interpreted_lithology_column"][2]
    ivs = result["intervals"]
    if not ivs:
        return
    ppf = result["depth_fit"]["px_per_ft"]
    y_start = ivs[0]["y_top_px"] if top_ft is None else min(
        (iv["y_top_px"] for iv in ivs if iv["base_ft"] > top_ft), default=ivs[0]["y_top_px"])
    y_start = max(0, y_start - int(10 * ppf))
    y_end = min(page.image.height, y_start + int(span_ft * ppf))
    scan = page.image.crop((dx0 - 10, y_start, lx1 + 10, y_end))
    panel_w = 420
    canvas = Image.new("RGB", (scan.width + panel_w, scan.height), "white")
    canvas.paste(scan, (0, 0))
    draw = ImageDraw.Draw(canvas)
    bx = scan.width + 10
    for iv in ivs:
        a, b = iv["y_top_px"] - y_start, iv["y_base_px"] - y_start
        if b < 0 or a > scan.height:
            continue
        col = PALETTE[iv["group"] % len(PALETTE)] if iv["group"] >= 0 else (200, 200, 200)
        draw.rectangle([bx, a, bx + 60, b], fill=col, outline=(0, 0, 0))
        draw.line([(0, a), (bx + 60, a)], fill=(255, 0, 0), width=1)
        label = f"{iv['top_ft']:.1f}-{iv['base_ft']:.1f}  {iv['lithology']}" + ("  (?)" if iv["ambiguous"] else "")
        if b - a > 12:
            draw.text((bx + 68, a + 2), label, fill=(0, 0, 0))
    canvas.save(out)


def run_file(image_path, boxes_path, out_dir) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    page = load_page(image_path, boxes_path)
    res = run_page(page)
    stem = Path(image_path).stem
    (out_dir / f"{stem}.json").write_text(json.dumps(res, indent=2))
    with open(out_dir / f"{stem}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["top_ft", "base_ft", "group", "lithology", "confidence", "ambiguous", "alternatives"])
        for iv in res["intervals"]:
            w.writerow([iv["top_ft"], iv["base_ft"], iv["group"], iv["lithology"], iv["confidence"], iv["ambiguous"], "|".join(iv["alternatives"])])
    render_overlay(page, res, out_dir / f"{stem}_overlay.png")
    return res
