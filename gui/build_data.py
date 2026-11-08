"""Build the data bundle for the Core Explorer GUI.

Reads the lithology results (one <log>.json per scan), the source scans and their bbox
annotations, and writes everything the browser needs into gui/data/:

    data/data.js            window.CORE = {logs: [...], points: {...}}   (loaded with a <script> tag,
                            so index.html also works when opened straight from disk)
    data/tiles/<id>/N.jpg   the scan body (depth column -> lithology column), downscaled, in tiles
    data/legend/<id>.jpg    the log's lithology legend crop

Usage (from the repo root; defaults read results/, dataset/images and dataset/bbox_annotation):
    python3 gui/build_data.py
    python3 gui/build_data.py --results PATH --images PATH --bboxes PATH   # elsewhere
    add --only <substring> to rebuild a few logs, --no-tiles to skip image work.
Needs numpy + Pillow only.
"""
import argparse, glob, json, os, re, time
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
HERE = Path(__file__).resolve().parent
TILE_H = 1024          # stored pixels per tile
MAX_W = 640            # stored width of the scan body
MAX_SCALE = 0.4

# ------------------------------------------------------------------ lithology categories
# Raw legend names vary a lot between logs; the GUI colours by a shared category so wells compare.
CATS = [  # (category, colour, keywords)   first match wins
    ("Annotation", "#d9d9d9", ["casing", "mud gain", "mud loss", "oil_", "gas_", "gas", "sample", "survey",
                               "core", "fit", "rft", "dry", "fresh water", "tar", "not available", "blank",
                               "eee", "others", "void", "available", "foraminifera", "glauconite"]),
    ("Halite / salt", "#e377c2", ["halite", "salt"]),
    ("Anhydrite / gypsum", "#b39ddb", ["anhyd", "gypsum"]),
    ("Coal", "#3a3a3a", ["coal", "lignite", "carbonaceous"]),
    ("Volcanic / igneous", "#c0392b", ["volcan", "tuff", "granite", "felsic", "dike", "igneous", "metamorph",
                                      "sericit", "cinder"]),
    ("Conglomerate / breccia", "#e67e22", ["conglom", "breccia"]),
    ("Chert", "#8e44ad", ["chert", "silice", "diatom"]),
    ("Dolomite", "#5dade2", ["dolo"]),
    ("Limestone / chalk", "#2e86c1", ["limestone", "chalk", "lime"]),
    ("Marl", "#76b7b2", ["marl", "mari"]),
    ("Sandstone", "#f4d03f", ["sand", "ss "]),
    ("Siltstone", "#d4a574", ["silt"]),
    ("Shale / clay", "#6b8e5a", ["shale", "clay", "argill", "mud"]),
]
CAT_COLOURS = {c: col for c, col, _ in CATS}
CAT_COLOURS["Unmatched group"] = "#9e9e9e"
CAT_COLOURS["Other"] = "#bdbdbd"


def normalise(iv):
    """Accept both results schemas (v1: top_ft/y_top_px/group, v2: top_depth/top_y/cluster_id)."""
    g = lambda *ks: next(iv[k] for k in ks if k in iv)
    return {"top_ft": g("top_ft", "top_depth"), "base_ft": g("base_ft", "bottom_depth"),
            "y_top_px": g("y_top_px", "top_y"), "y_base_px": g("y_base_px", "bottom_y"),
            "group": g("group", "cluster_id"), "lithology": iv["lithology"],
            "confidence": iv["confidence"], "ambiguous": iv["ambiguous"],
            "alternatives": iv.get("alternatives") or []}


def category(name):
    n = name.lower()
    if re.fullmatch(r"group_\d+", n) or n.startswith("unidentified"):
        return "Unmatched group"
    # rock words beat annotation words ("Medium Sand" in "Void eee Medium Sand")
    for cat, _, keys in CATS[1:]:
        if any(k in n for k in keys):
            return cat
    if any(k in n for k in CATS[0][2]):
        return "Annotation"
    return "Other"


def well_key(stem):
    """Logs from the same well share a key: first 10 digits of the US API number, else a name prefix."""
    m = re.match(r"(\d{12,14})", stem)
    if m:
        return m.group(1)[:10]
    m = re.match(r"0+\d_(d-\d+)", stem)
    if m:
        return m.group(1).upper()
    for pat in [r"(MC\d+_P\d+)", r"(Rio_[A-Za-z]+_\d+)", r"(Shell_[A-Z0-9]+_[A-Z0-9-]+)"]:
        m = re.match(pat, stem)
        if m:
            return m.group(1)
    return re.split(r"[ _]", stem)[0]


def short_name(stem):
    s = re.sub(r"^\d{12,14}_?", "", stem)
    s = re.sub(r"^0000000009_d-582_0000020435-0001_", "D-582 ", s)
    s = s.replace("_", " ").strip(" -")
    return s[:60]


# ------------------------------------------------------------------ image helpers
def open_scan(path):
    im = Image.open(path)
    bpp = {"1": 0.125, "L": 1, "P": 1}.get(im.mode, 3)
    if im.width * im.height * bpp < 6e8:
        im.load()  # decode once; huge scans are cropped strip by strip instead
    return im


def to_rgb(part):
    return part.convert("L").convert("RGB") if part.mode == "1" else part.convert("RGB")


def interval_features(rgb):
    """Texture + colour summary of one interval's lithology-column pixels (H x W x 3 uint8)."""
    if rgb.shape[0] < 2 or rgb.shape[1] < 4:
        return None
    a = rgb.astype(np.int16)
    lum = a.mean(2)
    ink = lum < 110
    sat = (a.max(2) - a.min(2)).astype(np.float32)
    coloured = (sat > 40) & ~ink
    ht = np.abs(np.diff(ink.astype(np.int8), axis=1)).mean()
    vt = np.abs(np.diff(ink.astype(np.int8), axis=0)).mean()
    rows_any = ink.any(1).mean()
    solid = (ink.mean(1) > 0.85).mean()
    fill = coloured.mean()
    if coloured.any():
        r, g, b = (a[coloured].mean(0) / 255.0)
    else:
        r = g = b = lum.mean() / 255.0
    return [ink.mean(), ht, vt, rows_any, solid, fill, r, g, b, lum.std() / 128.0]


FEATS = ["ink", "h_trans", "v_trans", "row_occ", "solid", "fill", "red", "green", "blue", "contrast"]


# ------------------------------------------------------------------ per log
def build_log(jpath, images, bboxes, out, tiles=True):
    res = json.load(open(jpath))
    stem = Path(res["image"]).stem
    lid = re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_")[:80]
    bb = json.load(open(Path(bboxes) / f"{stem}.json"))["bounding_boxes"]
    L, D = bb["interpreted_lithology_column"], bb["depth_column"]
    G = bb.get("lithology_legend")
    ivs = [normalise(i) for i in res["intervals"]]

    # depth <-> pixel row: least squares over every interval edge
    ys = np.array([v for i in ivs for v in (i["y_top_px"], i["y_base_px"])], float)
    fts = np.array([v for i in ivs for v in (i["top_ft"], i["base_ft"])], float)
    if len(ys) >= 2 and np.ptp(ys) > 0:
        b, a = np.polyfit(ys, fts, 1)
    else:
        b, a = 1.0 / res["depth_fit"]["px_per_ft"], 0.0

    x0 = max(0, min(D[0], L[0]) - 12)
    x1 = max(D[2], L[2]) + 12
    y0, y1 = L[1], L[3]
    s = min(MAX_SCALE, MAX_W / (x1 - x0))
    sw = int(round((x1 - x0) * s))
    sh = int(round((y1 - y0) * s))
    feats = [None] * len(ivs)

    im = open_scan(Path(images) / res["image"])
    x1 = min(x1, im.width)
    tdir = out / "tiles" / lid
    tdir.mkdir(parents=True, exist_ok=True)
    n_tiles = (sh + TILE_H - 1) // TILE_H
    src_tile = TILE_H / s
    order = sorted(range(len(ivs)), key=lambda k: ivs[k]["y_top_px"])
    k = 0
    for t in range(n_tiles):
        ys0 = y0 + int(t * src_tile)
        ys1 = min(y1, y0 + int((t + 1) * src_tile))
        if ys1 <= ys0:
            break
        part = to_rgb(im.crop((x0, ys0, x1, ys1)))
        if tiles:
            h = max(1, int(round((ys1 - ys0) * s)))
            part.resize((sw, h), Image.LANCZOS).save(tdir / f"{t}.jpg", quality=72, optimize=True)
        # interval features from this strip (intervals are split across strips; accumulate)
        arr = np.asarray(part)
        lx0, lx1 = L[0] - x0 + 4, L[2] - x0 - 4
        while k < len(order) and ivs[order[k]]["y_top_px"] < ys1:
            iv = ivs[order[k]]
            a0, a1 = max(iv["y_top_px"], ys0), min(iv["y_base_px"], ys1)
            if a1 > a0:
                f = interval_features(arr[a0 - ys0:a1 - ys0, lx0:lx1])
                if f is not None:
                    w = a1 - a0
                    prev = feats[order[k]]
                    feats[order[k]] = (np.array(f) * w + (prev[0] * prev[1] if prev else 0),
                                       w + (prev[1] if prev else 0))
                    feats[order[k]] = (feats[order[k]][0] / feats[order[k]][1], feats[order[k]][1])
            if iv["y_base_px"] <= ys1:
                k += 1
            else:
                break
    if G and tiles:
        (out / "legend").mkdir(exist_ok=True)
        lg = to_rgb(im.crop(tuple(G)))
        lg.thumbnail((900, 700))
        lg.save(out / "legend" / f"{lid}.jpg", quality=80)
    im.close()

    rows = []
    for i, iv in enumerate(ivs):
        rows.append({
            "t": round(iv["top_ft"], 2), "b": round(iv["base_ft"], 2),
            "y0": iv["y_top_px"], "y1": iv["y_base_px"],
            "g": iv["group"], "lith": iv["lithology"], "cat": category(iv["lithology"]),
            "c": iv["confidence"], "amb": bool(iv["ambiguous"]), "alt": iv.get("alternatives", []),
        })
    meta = {
        "id": lid, "stem": stem, "name": short_name(stem), "well": well_key(stem),
        "top": rows[0]["t"] if rows else 0, "base": rows[-1]["b"] if rows else 0,
        "fit": {"a": a, "b": b, **{k: v for k, v in res.get("depth_fit", {}).items() if k != "labels"}},
        "strip": {"x0": x0, "y0": y0, "w": sw, "h": sh, "s": s, "tiles": n_tiles, "tileH": TILE_H,
                  "lith": [round((L[0] - x0) * s), round((L[2] - x0) * s)],
                  "depth": [round((D[0] - x0) * s), round((D[2] - x0) * s)]},
        "legend": [l["name"] for l in res.get("legend", [])],
        "hasLegendImg": bool(G),
        "n_groups": res.get("n_groups"), "iv": rows,
        "warnings": res.get("warnings", []),
        "sameSymbols": res.get("identical_legend_symbols", []),
        "onGrid": (res.get("snap_grid") or {}).get("on_grid_fraction"),
    }
    return meta, [f[0] if f else None for f in feats]


# ------------------------------------------------------------------ embedding
def embed(logs, feats, k=4, seed=0):
    idx = [(li, ii) for li, fl in enumerate(feats) for ii, f in enumerate(fl) if f is not None]
    X = np.array([feats[li][ii] for li, ii in idx], float)
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    # PCA to 2D
    U, S, Vt = np.linalg.svd(X - X.mean(0), full_matrices=False)
    P = X @ Vt[:2].T
    P = (P - P.min(0)) / (np.ptp(P, 0) + 1e-9)
    # k nearest neighbours in feature space (blockwise), for the graph edges
    edges = []
    Xf = X.astype(np.float32)
    sq = (Xf ** 2).sum(1)
    for s0 in range(0, len(Xf), 1500):
        blk = Xf[s0:s0 + 1500]
        d = sq[s0:s0 + 1500, None] + sq[None, :] - 2 * blk @ Xf.T
        for r in range(len(blk)):
            d[r, s0 + r] = np.inf
        nn = np.argpartition(d, k, axis=1)[:, :k]
        for r, row in enumerate(nn):
            for j in row:
                if s0 + r < j:
                    edges.append((s0 + r, int(j)))
    var = (S ** 2) / (S ** 2).sum()
    return {
        "idx": [[li, ii] for li, ii in idx],
        "xy": np.round(P, 4).tolist(),
        "X": np.round(X, 2).tolist(),  # standardised features, for look-alike search in the browser
        "edges": edges,
        "explained": [round(float(v), 3) for v in var[:2]],
        "feats": FEATS,
        "loadings": np.round(Vt[:2], 3).tolist(),
    }


def first_dir(rel: str) -> Path:
    """The team repo keeps dataset/ at its root; older local checkouts had it under project/."""
    for c in (HERE / ".." / rel, HERE / "../project" / rel):
        if c.is_dir():
            return c
    return HERE / ".." / rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(HERE / "../results"))
    ap.add_argument("--images", default=str(first_dir("dataset/images")))
    ap.add_argument("--bboxes", default=str(first_dir("dataset/bbox_annotation")))
    ap.add_argument("--out", default=str(HERE / "data"))
    ap.add_argument("--only", default=None)
    ap.add_argument("--no-tiles", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "cache"
    cache.mkdir(exist_ok=True)

    logs, feats = [], []
    for jp in sorted(glob.glob(os.path.join(args.results, "*.json"))):
        stem = Path(jp).stem
        cp = cache / (re.sub(r"[^A-Za-z0-9]+", "_", stem)[:80] + ".json")
        t = time.time()
        fresh = cp.exists() and cp.stat().st_mtime >= os.path.getmtime(jp)
        if fresh and (args.only is None or args.only not in stem):
            m, f = json.load(open(cp))
        elif args.only is None or args.only in stem:
            try:
                m, f = build_log(jp, args.images, args.bboxes, out, tiles=not args.no_tiles)
            except Exception as e:  # keep going; one bad scan shouldn't stop the bundle
                print("FAILED", stem, e, flush=True)
                continue
            f = [list(map(float, x)) if x is not None else None for x in f]
            json.dump([m, f], open(cp, "w"))
            print(f"{time.time() - t:6.1f}s  {stem}", flush=True)
        else:
            continue
        logs.append(m)
        feats.append(f)

    pts = embed(logs, feats)
    bundle = {"logs": logs, "points": pts, "catColours": CAT_COLOURS,
              "catOrder": [c for c, _, _ in CATS[1:]] + ["Unmatched group", "Annotation", "Other"]}
    with open(out / "data.js", "w") as fh:
        fh.write("window.CORE = ")
        json.dump(bundle, fh, separators=(",", ":"))
        fh.write(";\n")
    print(f"wrote {len(logs)} logs, {len(pts['xy'])} points, {len(pts['edges'])} edges")


if __name__ == "__main__":
    main()
