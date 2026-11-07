#!/usr/bin/env python3
"""Run the lithology pipeline on one image or a whole dataset folder.

    python litho/run.py --dir dataset/ --out results/
    python litho/run.py IMAGE.tif BOXES.json --out results/

With --dir, images are searched for recursively (skipping bbox_vis/ and any
other visualisation folders), and each one is paired with the bounding-box JSON
whose "image_name" field names it (falling back to a JSON with the same stem).
"""
import argparse
import json
import sys
from pathlib import Path

# Make `lithoextract` importable however this script is launched.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lithoextract.pipeline import run_file  # noqa: E402

IMAGE_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
SKIP_DIRS = {"bbox_vis", "results", "__pycache__", ".git"}


def _skipped(path: Path, root: Path) -> bool:
    return any(part in SKIP_DIRS for part in path.relative_to(root).parts[:-1])


def find_pairs(root: Path) -> tuple[list[tuple[Path, Path]], list[Path]]:
    images = [p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS and not _skipped(p, root)]
    by_name: dict[str, Path] = {}
    by_stem: dict[str, Path] = {}
    for js in root.rglob("*.json"):
        if _skipped(js, root):
            continue
        try:
            name = json.loads(js.read_text()).get("image_name")
        except (json.JSONDecodeError, UnicodeDecodeError, AttributeError):
            continue
        if name:
            by_name[name] = js
            by_name[Path(name).stem] = js
        by_stem[js.stem] = js
    pairs, missing = [], []
    for img in sorted(images):
        js = by_name.get(img.name) or by_name.get(img.stem) or by_stem.get(img.stem)
        (pairs.append((img, js)) if js else missing.append(img))
    return pairs, missing


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?")
    ap.add_argument("boxes", nargs="?")
    ap.add_argument("--dir", help="dataset root (searched recursively)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--limit", type=int, default=None, help="only process the first N images")
    a = ap.parse_args()

    if a.dir:
        root = Path(a.dir)
        if not root.is_dir():
            sys.exit(f"--dir {root} is not a directory")
        pairs, missing = find_pairs(root)
        print(f"found {len(pairs)} image+box pairs under {root}/", flush=True)
        if missing:
            print(f"  {len(missing)} images had no matching bounding-box JSON, e.g. {missing[0]}", flush=True)
        if not pairs:
            sys.exit("nothing to do: check where the .json box files are and that their image_name matches")
    elif a.image and a.boxes:
        pairs = [(Path(a.image), Path(a.boxes))]
    else:
        ap.error("give IMAGE BOXES or --dir")

    if a.limit:
        pairs = pairs[: a.limit]
    ok = failed = 0
    for i, (img, js) in enumerate(pairs, 1):
        try:
            r = run_file(img, js, a.out)
            amb = sum(iv["ambiguous"] for iv in r["intervals"])
            grid = f"snap {r['snap_grid']['step_ft']:g} ft" if r["snap_grid"] else "no grid"
            print(f"[{i}/{len(pairs)}] {img.name[:60]:60} {len(r['intervals']):4d} intervals, "
                  f"{r['n_groups']} groups, {r['depth_fit']['labels_used']} depth labels "
                  f"({r['depth_fit']['labels_rejected']} rejected), {grid}, {amb} ambiguous, "
                  f"{r['timings_s']['total']}s", flush=True)
            for w in r["warnings"]:
                print(f"    warning: {w}", flush=True)
            ok += 1
        except Exception as e:  # keep going over a folder
            print(f"[{i}/{len(pairs)}] {img.name}: FAILED {type(e).__name__}: {e}", flush=True)
            failed += 1
    print(f"done: {ok} ok, {failed} failed -> {Path(a.out).resolve()}")


if __name__ == "__main__":
    main()
