# Core Explorer (G-Research: make the invisible visible)

A browser GUI for the lithology results: scan a new mud log (TIFF + bounding-box JSON), browse the team's 47 scanned logs, and see every well in 3D on a map of the Gulf of Mexico.

## Run it
From the repo root (the folder with `litho/`, `dataset/`, `results/`, `test/` and `gui/`):
```bash
pip install -r gui/requirements.txt
brew install tesseract           # or: sudo apt install tesseract-ocr / sudo port install tesseract tesseract-eng
python3 gui/server.py            # then open http://localhost:8765
```
On the **Scan a log** page, drop `test/images/<name>.tif` together with `test/annotations/<name>.json`
(select or drag both at once, in any order). The scanner runs the team's pipeline in `litho/` on the image as uploaded.
Nothing leaves your computer.

If `gui/data/` is missing, rebuild the library bundle from `results/` and `dataset/` with `python3 gui/build_data.py`
(about 5 minutes the first time). The server prints anything else that is missing when it starts.

## Scanning in more detail
1. The TIFF (or PNG/JPEG) is used pixel for pixel, at its own size and DPI, so the JSON's three boxes (`depth_column`,
   `interpreted_lithology_column`, `lithology_legend`, in image pixels) line up exactly. If the JSON's `width`/`height` don't
   match the image, the page says so. PDFs also work: each page is rendered at up to 300 DPI and the boxes are scaled to it.
2. Check the three boxes on the scan and drag an edge to adjust one if needed. Without a JSON, draw the boxes by hand, or load
   the JSON later with **Load boxes from a bounding-box JSON**.
3. Press **Scan this log**. It takes about 5 seconds to a minute or so for very long logs. You land on **Scanned logs**, a page for that file:
   the rock mix, every interval with its confidence, the legend it read, a CSV download, and a mini map with the nearest library wells.
   **Open the log report** shows it in the log viewer. **See it in 3D** shows it, in gold, among the other boreholes.
   The interval count is the same as running `python3 litho/run.py` on the same image and JSON.

Where the well goes on the map: if the file name names a library well, the scan is put at that well. Otherwise a lease block in the name (e.g. `GC 562`) is
turned into a position on the block grid, which is usually within a block (about 5 km). An API number gives the area only. If none of these work, type the block in
or drop a pin by hand.
Uploads live in `gui/uploads/<id>/`, with the source file, the rendered pages and the raw pipeline output (JSON, CSV, overlay). They are listed for the GUI in `data/uploads.js`.
Server options: `--port`, `--litho PATH` (default: `litho/` next to `gui/`), `--no-browser`.

## Rebuilding the library bundle
```bash
python3 gui/build_data.py        # reads results/, dataset/images, dataset/bbox_annotation; writes gui/data/
```
A log is only reprocessed when its results JSON is newer than its cache entry; add `--no-tiles` to skip re-cutting image tiles,
and `--results` / `--images` / `--bboxes` to read from elsewhere. Without the server, `gui/index.html` can be opened straight
from disk to browse (but not scan).

## Views
* **Scan a log** / **Scanned logs**: see above.
* **Map**: 3D boreholes (MapLibre + deck.gl, open source and bundled in `vendor/`) with each log drawn underground at its real depth, or a 2D map (Leaflet).
* **Library**: every log as a card with its file name decoded (area, block, lease, scale, MD/TVD…), with filters.
* **Log**: the real scan (depth column to lithology column) next to the extracted column. Blocks are coloured by rock type,
  faded by confidence, and hatched when ambiguous. There is also a confidence trace and depth ticks from the depth fit.
  Hover to read an interval; click to pin it and see its alternatives and its look-alikes in other logs. Use the up/down
  arrow keys to step through intervals. The minimap on the left lets you jump along the log.
* **Correlate**: the ticked logs side by side on a shared depth axis, or with "align tops" each from its own top.
  Click any block to open it in the log viewer.
* **Interval map** (3D, rotatable): every interval from every log, laid out by the texture and colour of its patch of scan
  (top 3 principal components of 10 pixel features, computed in the browser), with links to its 4 nearest neighbours. Colour it by rock type, well or confidence.
  Click a dot to find look-alikes in other logs.
* **Field**: headline numbers, then the lithology mix and ambiguous share for every log and well.

Clicking a rock type in any legend highlights it everywhere.

## Data notes
* Raw legend names are mapped to shared categories (`CATS` in `build_data.py`) so that wells can be compared.
  The raw name is always shown in tooltips.
* `group_N` (no legend match) shows as "Unmatched group". Annotation symbols (casing, mud gain, oil/gas shows…) show as "Annotation".
* Logs are grouped into wells by the first 10 digits of the API number, or by a name prefix.
* "depth uncertain" flags logs whose depth fit has a negative top, a residual above 0.4 ft, or fewer than 4 labels.
