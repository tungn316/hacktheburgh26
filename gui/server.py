#!/usr/bin/env python3
"""Core Explorer local scanner: serves the GUI and runs the team's lithology pipeline on uploaded logs.

    python3 server.py                  # then open http://localhost:8765
    python3 server.py --litho ../litho # point at another checkout of the pipeline

Upload flow (all local, nothing leaves the machine):
  POST /api/upload?name=FILE.pdf   raw file bytes   -> renders the PDF page(s) to PNG, returns a preview
  POST /api/scan/<id>              {"page": 0, "boxes": {...}}   -> runs litho/run.py's pipeline on that page,
                                   turns the result into GUI data and adds it to data/uploads.js
  POST /api/locate/<id>            {"lat": .., "lon": .., "precision": "..", "desc": ".."}   -> saves a map position
  POST /api/delete/<id>            removes an uploaded scan from the GUI
  GET  /api/status                 which dependencies are present
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from PIL import Image

Image.MAX_IMAGE_PIXELS = None
# Tesseract's OpenMP threads can stall for minutes when several OCR calls compete for the CPU;
# one thread per call is just as fast for the small crops the pipeline sends it.
os.environ.setdefault("OMP_THREAD_LIMIT", "1")
HERE = Path(__file__).resolve().parent
UPLOADS = HERE / "uploads"            # working files per upload (source, page images, raw pipeline output)
DATA = HERE / "data"                  # what the browser loads
MAX_PIXELS = 150e6                    # cap on a rendered page (RGB) so very long logs stay within memory
PREVIEW_W = 1100                      # preview width shown for drawing boxes
PREVIEW_MAX_H = 60000                 # JPEG tops out at 65,535 px; some mud logs are 140,000 px tall
LOCK = threading.Lock()               # one scan at a time; the pipeline is CPU and memory heavy
IMAGE_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}

sys.path.insert(0, str(HERE))
import build_data  # noqa: E402  (same folder: tiles, features, categories)


# ------------------------------------------------------------------ dependencies
def find_litho(arg: str | None) -> Path | None:
    cands = [Path(arg)] if arg else [HERE / "../litho", HERE / "../project/litho", *HERE.parent.glob("hacktheburgh26*/litho")]
    for c in cands:
        if (c / "lithoextract" / "pipeline.py").exists():
            return c.resolve()
    return None


def status(litho: Path | None) -> dict:
    s = {"litho": str(litho) if litho else None, "missing": []}
    if not litho:
        s["missing"].append("the team's pipeline folder (litho/ with lithoextract/ inside); start with --litho PATH")
    for mod, pip in [("numpy", "numpy"), ("cv2", "opencv-python-headless"), ("pytesseract", "pytesseract")]:
        try:
            __import__(mod)
        except ImportError:
            s["missing"].append(f"Python package {pip}  (pip3 install {pip})")
    if not shutil.which("tesseract"):
        s["missing"].append("the tesseract OCR program  (brew install tesseract, or apt install tesseract-ocr)")
    s["pdf"] = pdf_backend()
    if not s["pdf"]:
        s["missing"].append("a PDF renderer: pip3 install pymupdf  (or brew install poppler)")
    return s


def pdf_backend() -> str | None:
    try:
        import pymupdf  # noqa: F401
        return "pymupdf"
    except ImportError:
        try:
            import fitz  # noqa: F401
            return "fitz"
        except ImportError:
            return "pdftoppm" if shutil.which("pdftoppm") else None


# ------------------------------------------------------------------ PDF / image -> page PNGs
def render_pdf(src: Path, out: Path) -> list[dict]:
    """Every page to PNG at up to 300 DPI, lowered for very long pages so they fit in memory."""
    be = pdf_backend()
    pages = []
    if be in ("pymupdf", "fitz"):
        mod = __import__(be)
        doc = mod.open(str(src))
        for i, pg in enumerate(doc):
            w_in, h_in = pg.rect.width / 72, pg.rect.height / 72
            dpi = int(min(300, (MAX_PIXELS / max(1e-6, w_in * h_in)) ** 0.5))
            pix = pg.get_pixmap(dpi=dpi, alpha=False)
            p = out / f"page{i + 1}.png"
            Image.frombytes("RGB", (pix.width, pix.height), pix.samples).save(p, dpi=(dpi, dpi))
            pages.append({"file": p.name, "dpi": dpi, "w": pix.width, "h": pix.height})
    elif be == "pdftoppm":
        info = subprocess.run(["pdfinfo", str(src)], capture_output=True, text=True).stdout
        m = re.search(r"Page size:\s+([\d.]+) x ([\d.]+)", info)
        w_in, h_in = (float(m.group(1)) / 72, float(m.group(2)) / 72) if m else (8.5, 11)
        dpi = int(min(300, (MAX_PIXELS / (w_in * h_in)) ** 0.5))
        subprocess.run(["pdftoppm", "-png", "-r", str(dpi), str(src), str(out / "page")], check=True)
        for i, p in enumerate(sorted(out.glob("page-*.png"), key=lambda q: int(re.findall(r"\d+", q.stem)[-1]))):
            q = out / f"page{i + 1}.png"
            p.rename(q)
            with Image.open(q) as im:
                im.save(q, dpi=(dpi, dpi))
                pages.append({"file": q.name, "dpi": dpi, "w": im.width, "h": im.height})
    else:
        raise RuntimeError("No PDF renderer installed: run  pip3 install pymupdf")
    if not pages:
        raise RuntimeError("That PDF has no pages")
    return pages


def link_or_copy(src: Path, dst: Path):
    """Byte-for-byte copy of an uploaded image (a hard link when possible, to spare disk on 30 MB scans)."""
    dst.unlink(missing_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def image_page(src: Path, out: Path) -> list[dict]:
    """TIFF/PNG/JPEG scans are used as uploaded: same bytes, pixels and DPI, so the JSON's boxes line up 1:1."""
    try:
        with Image.open(src) as im:
            dpi = float((im.info.get("dpi") or (0, 0))[0] or 0)
            w, h = im.size
    except Exception as e:
        raise RuntimeError(f"Couldn't read that image ({e}). Is it a valid TIFF/PNG/JPEG?")
    p = out / ("page1" + src.suffix.lower())
    link_or_copy(src, p)
    return [{"file": p.name, "dpi": round(dpi) or None, "w": w, "h": h}]


def preview(page_path: Path, dst: Path) -> float:
    """A small JPEG to draw boxes on; boxes are always kept in the full image's pixels."""
    with Image.open(page_path) as im:
        s = min(1.0, PREVIEW_W / im.width, PREVIEW_MAX_H / im.height)
        size = (max(1, round(im.width * s)), max(1, round(im.height * s)))
        if im.mode == "1":
            im = im.convert("L")
        elif im.mode not in ("L", "RGB"):
            im = im.convert("RGB")
        f = int(1 / s) if s < 1 else 1
        if f >= 2:  # cheap integer shrink first: keeps memory and time low on huge scans
            im = im.reduce(f)
        im.resize(size, Image.LANCZOS).convert("RGB").save(dst, quality=80)
    return s


# ------------------------------------------------------------------ helpers
def slug(name: str) -> str:
    return "up_" + re.sub(r"[^A-Za-z0-9]+", "_", Path(name).stem).strip("_")[:70]


def read_json(p: Path, default=None):
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return default


def write_uploads_js():
    """data/uploads.js = every finished upload, newest first; index.html merges these into the library."""
    metas = []
    for d in sorted(UPLOADS.glob("up_*")):
        m = read_json(d / "gui.json")
        if m:
            metas.append(m)
    metas.sort(key=lambda m: -m.get("uploadedAt", 0))
    DATA.mkdir(exist_ok=True)
    (DATA / "uploads.js").write_text("window.UPLOADS = " + json.dumps(metas, separators=(",", ":")) + ";\n")


def run_scan(litho: Path, uid: str, page_no: int, boxes: dict, json_size=None) -> dict:
    job = UPLOADS / uid
    info = read_json(job / "info.json")
    if not info:
        raise RuntimeError("Unknown upload; please upload the file again")
    pg = info["pages"][page_no]
    stem = Path(info["name"]).stem
    # The pipeline and the GUI builder name everything after the image's stem, so give the page the original name.
    work = job / "work"
    shutil.rmtree(work, ignore_errors=True)
    (work / "images").mkdir(parents=True)
    (work / "bbox").mkdir()
    img = work / "images" / (stem + Path(pg["file"]).suffix)
    link_or_copy(job / pg["file"], img)
    if json_size and info.get("image") and [int(v) for v in json_size] != [pg["w"], pg["h"]]:
        raise RuntimeError(f"This JSON is for a {json_size[0]} x {json_size[1]} image but the {Path(info['name']).suffix.lstrip('.').upper()} "
                           f"is {pg['w']} x {pg['h']}. Use the JSON made for this image.")
    keys = ["depth_column", "interpreted_lithology_column", "lithology_legend"]
    bb = {k: [int(round(v)) for v in boxes[k]] for k in keys}
    for k, (x0, y0, x1, y1) in bb.items():
        bb[k] = [max(0, min(x0, x1)), max(0, min(y0, y1)), min(pg["w"], max(x0, x1)), min(pg["h"], max(y0, y1))]
        if bb[k][2] - bb[k][0] < 4 or bb[k][3] - bb[k][1] < 4:
            raise RuntimeError(f"The {k.replace('_', ' ')} box is too small")
    (work / "bbox" / f"{stem}.json").write_text(json.dumps(
        {"image_name": img.name, "width": pg["w"], "height": pg["h"], "bounding_boxes": bb}, indent=2))

    if str(litho) not in sys.path:
        sys.path.insert(0, str(litho))
    from lithoextract.pipeline import run_file  # the team's code, unchanged

    t0 = time.time()
    res_dir = work / "results"
    res = run_file(img, work / "bbox" / f"{stem}.json", res_dir)
    t_pipe = time.time() - t0

    build = work / "gui"
    meta, feats = build_data.build_log(res_dir / f"{stem}.json", work / "images", work / "bbox", build)
    lid = meta["id"]
    # Move tiles and legend crop into the GUI's data folder under the upload's own id.
    for sub, name in [("tiles", lid), ("legend", lid + ".jpg")]:
        src = build / sub / name
        dst = DATA / sub / (uid + (".jpg" if sub == "legend" else ""))
        if dst.exists():
            shutil.rmtree(dst) if dst.is_dir() else dst.unlink()
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
    old = read_json(job / "gui.json") or {}
    meta.update({
        "id": uid, "upload": True, "well": uid, "srcWell": meta["well"],
        "uploadedAt": info["uploadedAt"], "file": info["name"], "page": page_no + 1, "pages": len(info["pages"]),
        "dpi": pg["dpi"], "boxes": bb, "pipelineSeconds": round(t_pipe, 1),
        "overlay": f"uploads/{uid}/work/results/{stem}_overlay.png" if (res_dir / f"{stem}_overlay.png").exists() else None,
        "csv": f"uploads/{uid}/work/results/{stem}.csv",
        "loc": old.get("loc"),
    })
    (job / "gui.json").write_text(json.dumps(meta))
    shutil.rmtree(build, ignore_errors=True)
    write_uploads_js()
    return meta


# ------------------------------------------------------------------ HTTP
class Handler(SimpleHTTPRequestHandler):
    litho: Path | None = None

    def __init__(self, *a, **k):
        super().__init__(*a, directory=str(HERE), **k)

    def log_message(self, fmt, *args):  # quieter: only API calls
        if args and isinstance(args[0], str) and "/api/" in args[0]:
            sys.stderr.write("  " + fmt % args + "\n")

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def send_json(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/api/status":
            return self.send_json(status(self.litho))
        if u.path == "/data/uploads.js" and not (DATA / "uploads.js").exists():
            write_uploads_js()
        return super().do_GET()

    def body(self) -> bytes:
        n = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(n) if n else b""

    def do_POST(self):
        u = urlparse(self.path)
        parts = [unquote(p) for p in u.path.strip("/").split("/")]
        try:
            if parts[:2] == ["api", "upload"]:
                return self.send_json(self.upload(parse_qs(u.query).get("name", ["upload.pdf"])[0]))
            if parts[:2] == ["api", "scan"] and len(parts) == 3:
                req = json.loads(self.body() or b"{}")
                if not self.litho:
                    raise RuntimeError("The pipeline folder (litho/) was not found. Restart with  python3 server.py --litho PATH")
                if not LOCK.acquire(timeout=1):
                    raise RuntimeError("Another scan is still running; try again in a moment")
                try:
                    return self.send_json(run_scan(self.litho, safe_id(parts[2]), int(req.get("page", 0)), req["boxes"], req.get("jsonSize")))
                finally:
                    LOCK.release()
            if parts[:2] == ["api", "locate"] and len(parts) == 3:
                uid, req = safe_id(parts[2]), json.loads(self.body() or b"{}")
                m = read_json(UPLOADS / uid / "gui.json")
                if not m:
                    raise RuntimeError("Scan not found")
                m["loc"] = None if req.get("clear") else [float(req["lat"]), float(req["lon"]), str(req.get("precision", "user")), str(req.get("desc", ""))[:200]]
                (UPLOADS / uid / "gui.json").write_text(json.dumps(m))
                write_uploads_js()
                return self.send_json({"ok": True, "loc": m["loc"]})
            if parts[:2] == ["api", "delete"] and len(parts) == 3:
                uid = safe_id(parts[2])
                shutil.rmtree(UPLOADS / uid, ignore_errors=True)
                shutil.rmtree(DATA / "tiles" / uid, ignore_errors=True)
                (DATA / "legend" / f"{uid}.jpg").unlink(missing_ok=True)
                write_uploads_js()
                return self.send_json({"ok": True})
            self.send_json({"error": "unknown endpoint"}, 404)
        except Exception as e:  # report to the page in plain words
            traceback.print_exc()
            self.send_json({"error": f"{type(e).__name__}: {e}" if not isinstance(e, RuntimeError) else str(e)}, 400)

    def upload(self, name: str) -> dict:
        name = Path(name).name or "upload.pdf"
        ext = Path(name).suffix.lower()
        if ext != ".pdf" and ext not in IMAGE_EXTS:
            raise RuntimeError("Please upload the log image (.tif/.tiff, or PNG/JPEG/PDF)")
        data = self.body()
        if not data:
            raise RuntimeError("The file was empty")
        uid = slug(name)
        job = UPLOADS / uid
        shutil.rmtree(job, ignore_errors=True)
        job.mkdir(parents=True)
        src = job / ("source" + ext)
        src.write_bytes(data)
        del data
        pages = render_pdf(src, job) if ext == ".pdf" else image_page(src, job)
        for i, p in enumerate(pages):
            p["preview"] = f"uploads/{uid}/preview{i + 1}.jpg"
            p["previewScale"] = preview(job / p["file"], job / f"preview{i + 1}.jpg")
        info = {"id": uid, "name": name, "image": ext != ".pdf", "pages": pages, "uploadedAt": int(time.time() * 1000)}
        (job / "info.json").write_text(json.dumps(info))
        return info


def safe_id(s: str) -> str:
    if not re.fullmatch(r"up_[A-Za-z0-9_]{1,80}", s):
        raise RuntimeError("Bad scan id")
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--litho", default=None, help="folder holding lithoextract/ (default: ../litho)")
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    Handler.litho = find_litho(a.litho)
    UPLOADS.mkdir(exist_ok=True)
    write_uploads_js()
    s = status(Handler.litho)
    print(f"Core Explorer scanner\n  pipeline: {s['litho'] or 'NOT FOUND'}\n  PDF renderer: {s['pdf'] or 'none'}")
    if not (DATA / "data.js").exists():
        s["missing"].append("the library bundle gui/data/data.js  (run  python3 gui/build_data.py)")
    print("  ready: nothing missing" if not s["missing"] else "  MISSING: " + "; ".join(s["missing"]))
    url = f"http://localhost:{a.port}/"
    print(f"  open {url}   (Ctrl+C to stop)")
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    if not a.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
