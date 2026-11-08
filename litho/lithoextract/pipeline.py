"""End to end: image + boxes -> depth-registered, legend-labelled lithology intervals.

Output CSV (one row per interval, contiguous, full-image pixel rows):
    top_depth, bottom_depth, lithology, cluster_id, top_y, bottom_y
The JSON beside it keeps the extras: confidence, ambiguity, alternatives,
the depth fit, the legend, the snapping grid, warnings and timings.
"""
from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .depth import DepthFit, calibrate_depth
from .features import interior
from .io import Page, load_page
from .legend import Swatch, ink_similarity, parse_legend
from .overlay import render_overlays
from .segment import Column, group_segments, segment_column
from .shape import shape_descriptor, shape_distance
from .snap import Grid, find_grid
from .tiles import name_pieces

AMBIGUOUS_MARGIN = 0.10  # best and runner-up legend matches closer than this -> flag as ambiguous
IDENTICAL_INK = 0.9  # legend swatches whose ink matches this well are the same symbol...
IDENTICAL_SHAPE = 0.15  # ...provided their stroke-shape descriptors agree too
CSV_FIELDS = ["top_depth", "bottom_depth", "lithology", "cluster_id", "top_y", "bottom_y"]

# Rough order of how common each rock is in the logs we see; breaks ties between identical symbols.
COMMONNESS = ["shale", "sandstone", "siltstone", "limestone", "claystone", "mudstone", "clay", "sand", "marl",
              "dolomite", "anhydrite", "salt", "halite", "gypsum", "coal", "lignite", "chert", "conglomerate",
              "tuff", "volcanic", "igneous", "cement"]


def _commonness(name: str) -> int:
    words = name.lower().replace("-", " ").split()
    ranks = [COMMONNESS.index(w) for w in words if w in COMMONNESS]
    return min(ranks) if ranks else len(COMMONNESS)


@dataclass
class Interval:
    y0: int  # absolute pixel rows, [y0, y1)
    y1: int
    cluster_id: int
    lithology: str
    confidence: float
    alternatives: list[str]
    ambiguous: bool
    blank: bool = False
    top_depth: float = 0.0
    bottom_depth: float = 0.0
    snapped: list[bool] = field(default_factory=lambda: [False, False])


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
    if not parts:
        return np.zeros((2, 1))
    rgb = np.concatenate(parts)
    # Two views: as drawn, and without tile-edge lines (see shape_descriptor). Each legend symbol is
    # compared with the closer view, so a symbol that really is lines still matches with them kept.
    return np.stack([shape_descriptor(rgb), shape_descriptor(rgb, drop_rules=True)])


def _twins(swatches: list[Swatch]) -> dict[str, set[str]]:
    """Legend names whose symbols are drawn identically (e.g. Clay and Shale on some logs)."""
    out: dict[str, set[str]] = {}
    for i, a in enumerate(swatches):
        for b in swatches[i + 1 :]:
            if a.name == b.name:
                continue
            if shape_distance(a.shape, b.shape) < IDENTICAL_SHAPE and ink_similarity(a.ink, b.ink) >= IDENTICAL_INK:
                out.setdefault(a.name, set()).add(b.name)
                out.setdefault(b.name, set()).add(a.name)
    return out


def _label_groups(rgb, segs, n_groups, swatches):
    candidates = [s for s in swatches if not s.is_white]
    twins = _twins(candidates)
    labels: dict[int, tuple[str, float, list[str], bool]] = {}
    for g in range(n_groups):
        gd = _group_shape(rgb, segs, g)
        ranked = sorted((min(shape_distance(v, s.shape) for v in gd), s.name) for s in candidates)
        if len(ranked) < 2:  # with 0-1 legend symbols there is nothing to choose between
            labels[g] = (f"Unidentified (cluster {g})", 0.0, [n for _, n in ranked], True)
            continue
        d1, n1 = ranked[0]
        tied = {n1} | twins.get(n1, set())
        # Confidence is measured against the best *distinguishable* alternative.
        others = [d for d, n in ranked if n not in tied]
        d2 = others[0] if others else d1 + 1.0
        margin = d2 - d1
        conf = float(np.clip(margin / (d1 + margin + 1e-6), 0, 1))
        name = min(tied, key=lambda n: (_commonness(n), n))  # identical symbols: the more common rock
        alts = sorted(tied - {name}) + [n for d, n in ranked[1:4] if n not in tied and d - d1 < 0.5]
        labels[g] = (name, round(conf, 3), alts, margin < AMBIGUOUS_MARGIN or len(tied) > 1)
    return labels, twins


def _merge_into(ivs: list[Interval], i: int, j: int) -> None:
    """Interval i is absorbed by its neighbour j; j keeps its label and grows over i."""
    a, b = ivs[i], ivs[j]
    if j < i:
        b.y1, b.bottom_depth, b.snapped[1] = a.y1, a.bottom_depth, a.snapped[1]
    else:
        b.y0, b.top_depth, b.snapped[0] = a.y0, a.top_depth, a.snapped[0]
    del ivs[i]


def _merge_same_name(ivs: list[Interval]) -> None:
    i = 0
    while i + 1 < len(ivs):
        a, b = ivs[i], ivs[i + 1]
        if a.lithology == b.lithology:
            keep = a if a.y1 - a.y0 >= b.y1 - b.y0 else b  # the longer piece's attributes win
            merged = Interval(a.y0, b.y1, keep.cluster_id, keep.lithology, keep.confidence, keep.alternatives,
                              keep.ambiguous, keep.blank, a.top_depth, b.bottom_depth, [a.snapped[0], b.snapped[1]])
            ivs[i : i + 2] = [merged]
        else:
            i += 1


def _snap(ivs: list[Interval], fit: DepthFit, grid: Grid, ylo: int, yhi: int) -> int:
    """Move every boundary that is near a grid line onto it. Returns how many moved."""
    moved = 0
    for k in range(len(ivs) + 1):
        d = ivs[k].top_depth if k < len(ivs) else ivs[-1].bottom_depth
        if not grid.on(d):
            continue
        gd = grid.nearest(d)
        y = int(round(fit.row(gd)))
        lo = ivs[k - 1].y0 + 1 if k > 0 else ylo
        hi = ivs[k].y1 - 1 if k < len(ivs) else yhi
        if not lo <= y <= hi:
            continue
        moved += y != (ivs[k].y0 if k < len(ivs) else ivs[-1].y1)
        if k < len(ivs):
            ivs[k].y0, ivs[k].top_depth, ivs[k].snapped[0] = y, gd, True
        if k > 0:
            ivs[k - 1].y1, ivs[k - 1].bottom_depth, ivs[k - 1].snapped[1] = y, gd, True
    return moved


def _fix_offgrid_pieces(ivs: list[Interval], col: Column, fit: DepthFit, y_off: int, grid: Grid) -> tuple[int, int]:
    """Short pieces (< one grid step) with an off-grid internal edge.

    Shortest first, the off-grid edge is moved to its nearest grid line if both intervals on
    either side of it survive (a bed whose faded top was detected a few feet low). If moving it
    would erase the piece, the piece is a faded or smudged patch of a neighbouring rock (e.g. a
    3 ft "sandstone" at the faded base of a siltstone) and joins its most similar neighbour.
    Returns (edges moved, pieces absorbed).
    """
    def desc(v: Interval) -> np.ndarray:
        m = (v.y1 - v.y0) // 6
        return col.desc(v.y0 - y_off + m, v.y1 - y_off - m)

    def offgrid_edges(i: int) -> list[int]:  # 0 = top edge, 1 = bottom edge (internal edges only)
        return [e for e, ok in ((0, i > 0), (1, i < len(ivs) - 1)) if ok and not ivs[i].snapped[e]]

    moved = absorbed = 0
    while len(ivs) > 1:
        cands = [i for i, iv in enumerate(ivs) if iv.bottom_depth - iv.top_depth < grid.step and offgrid_edges(i)]
        if not cands:
            break
        i = min(cands, key=lambda i: ivs[i].y1 - ivs[i].y0)  # a 3 ft sliver before the 7 ft bed it was cut from
        iv = ivs[i]
        relocated = False
        for e in offgrid_edges(i):
            gd = grid.nearest(iv.bottom_depth if e else iv.top_depth)
            y = int(round(fit.row(gd)))
            upper, lower = (iv, ivs[i + 1]) if e else (ivs[i - 1], iv)
            if upper.y0 < y < lower.y1 and upper.top_depth < gd < lower.bottom_depth:
                upper.y1, upper.bottom_depth, upper.snapped[1] = y, gd, True
                lower.y0, lower.top_depth, lower.snapped[0] = y, gd, True
                moved += 1
                relocated = True
                break
        if relocated:
            continue
        me = desc(iv)
        nb = [j for j in (i - 1, i + 1) if 0 <= j < len(ivs)]
        j = min(nb, key=lambda j: float(np.linalg.norm(desc(ivs[j]) - me)))
        _merge_into(ivs, i, j)
        absorbed += 1
    return moved, absorbed


def run_page(page: Page) -> dict:
    timings: dict[str, float] = {}
    t0 = t = time.time()

    def lap(name: str) -> None:
        nonlocal t
        now = time.time()
        timings[name] = round(now - t, 2)
        t = now

    fit = calibrate_depth(page)
    ppf = fit.px_per_ft
    lap("depth")

    x0, y0, x1, y1 = page.boxes["interpreted_lithology_column"]
    rgb = interior(page.crop((x0, y0, x1, y1)))
    segs, col = segment_column(rgb, ppf)
    n_groups = group_segments(segs)
    lap("segment")

    swatches = parse_legend(page, ppf)
    lap("legend")

    group_label, twins = _label_groups(rgb, segs, n_groups, swatches)
    white = [s for s in swatches if s.is_white]
    blank_cluster = n_groups  # blank stretches inside the log get a cluster of their own

    # Where the legend's own drawing can be found in the column, name each piece from that.
    piece_label = name_pieces(rgb, segs, swatches, twins, _commonness, lambda s: group_label[s.group][3])
    matching = "legend drawing found in column" if piece_label is not None else "stroke fingerprints"

    ivs: list[Interval] = []
    for k, s in enumerate(segs):
        if s.blank:
            if white:
                name, conf, alts, amb = white[0].name, 0.8, [], False
            else:
                name, conf, alts, amb = "Blank (no pattern)", 0.5, [], True
            g = blank_cluster
        else:
            g = s.group
            name, conf, alts, amb = group_label[g]
            if piece_label and k in piece_label:
                name, conf, alts, amb = piece_label[k]
        ivs.append(Interval(y0 + s.y0, y0 + s.y1, g, name, conf, list(alts), amb, s.blank))
    if piece_label is not None:  # pieces were named one by one: a cluster is now simply one rock
        cid: dict[str, int] = {}
        for iv in ivs:
            if not iv.blank:
                iv.cluster_id = cid.setdefault(iv.lithology, len(cid))
        n_groups = len(cid)
        for iv in ivs:
            if iv.blank:
                iv.cluster_id = n_groups
    # Blank stretches at either end are outside the logged range.
    while ivs and ivs[0].blank:
        ivs.pop(0)
    while ivs and ivs[-1].blank:
        ivs.pop()
    if not ivs:
        raise RuntimeError("no patterned intervals found in the lithology column")
    for iv in ivs:
        iv.top_depth, iv.bottom_depth = fit.depth(iv.y0), fit.depth(iv.y1)
    _merge_same_name(ivs)

    grid = find_grid([iv.top_depth for iv in ivs[1:]], ppf)
    n_snapped = n_relocated = n_absorbed = 0
    if grid:
        n_snapped = _snap(ivs, fit, grid, y0, y1)
        n_relocated, n_absorbed = _fix_offgrid_pieces(ivs, col, fit, y0, grid)
        _merge_same_name(ivs)
        if n_absorbed:  # an absorbed sliver can leave an edge next to a grid line that is free again
            n_snapped += _snap(ivs, fit, grid, y0, y1)
    for iv in ivs:
        iv.top_depth = iv.top_depth if iv.snapped[0] else round(float(fit.depth(iv.y0)), 2)
        iv.bottom_depth = iv.bottom_depth if iv.snapped[1] else round(float(fit.depth(iv.y1)), 2)
    lap("intervals")

    warnings = list(page.warnings) + fit.warnings
    timings["total"] = round(time.time() - t0, 2)
    return {
        "image": page.path.name,
        "image_size": [page.width, page.height],
        "depth_fit": {
            "px_per_ft": round(ppf, 4),
            "labels_used": len(fit.ys),
            "labels_read": fit.n_read,
            "labels_rejected": fit.n_rejected,
            "mean_residual_ft": round(fit.residual_ft, 3),
            "label_rotation_deg": fit.rotation,
            "chunks_ocrd": f"{fit.chunks_read}/{fit.chunks_total}",
            "labels": [[round(y, 1), d] for y, d in fit.points],
        },
        "legend": [{"name": s.name, "box": list(s.box), "white": s.is_white} for s in swatches],
        "identical_legend_symbols": sorted({tuple(sorted({k} | v)) for k, v in twins.items()}),
        "n_groups": n_groups,
        "naming": matching,
        "snap_grid": None if grid is None else {"step_ft": grid.step, "tolerance_ft": grid.tol,
                                                "on_grid_fraction": round(grid.on_grid_frac, 3),
                                                "boundaries_snapped": n_snapped, "short_piece_edges_moved_to_grid": n_relocated,
                                                "slivers_absorbed": n_absorbed},
        "warnings": warnings,
        "timings_s": timings,
        "intervals": [
            {"top_depth": iv.top_depth, "bottom_depth": iv.bottom_depth, "lithology": iv.lithology,
             "cluster_id": iv.cluster_id, "top_y": iv.y0, "bottom_y": iv.y1, "confidence": iv.confidence,
             "ambiguous": iv.ambiguous, "alternatives": iv.alternatives}
            for iv in ivs
        ],
    }


def check_intervals(result: dict, page: Page) -> None:
    """The checks the example notebook makes on a results CSV, plus depth order."""
    ivs = result["intervals"]
    box = page.boxes["interpreted_lithology_column"]
    assert ivs, "no intervals"
    assert all(a["bottom_y"] == b["top_y"] for a, b in zip(ivs, ivs[1:])), "gap or overlap in pixel rows"
    assert all(a["bottom_depth"] == b["top_depth"] for a, b in zip(ivs, ivs[1:])), "gap or overlap in depth"
    assert all(iv["top_y"] < iv["bottom_y"] and iv["top_depth"] < iv["bottom_depth"] for iv in ivs), "empty interval"
    assert box[1] <= ivs[0]["top_y"] and ivs[-1]["bottom_y"] <= box[3], "intervals leave the lithology column"


def run_file(image_path, boxes_path, out_dir) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    page = load_page(image_path, boxes_path)
    res = run_page(page)
    check_intervals(res, page)
    stem = Path(image_path).stem
    (out_dir / f"{stem}.json").write_text(json.dumps(res, indent=2))
    with open(out_dir / f"{stem}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(res["intervals"])
    render_overlays(page, res, out_dir, stem)
    return res
