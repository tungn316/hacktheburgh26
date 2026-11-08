2 lithoextract: lithology intervals from scanned well logs

```
pip install -r requirements.txt            # + tesseract binary
python run.py IMAGE.tif BOXES.json --out results/
python run.py --dir data/ --out results/   # every IMAGE.tif with a matching IMAGE.json
```

Outputs per image:
* `IMAGE.csv`: exactly `top_depth, bottom_depth, lithology, cluster_id, top_y, bottom_y`, one row per
  interval, contiguous (each `bottom_y` is the next `top_y`), rows in full-image pixels, the format the
  example notebook checks. Every run asserts those checks before writing.
* `IMAGE.json`: the same intervals plus `confidence`, `ambiguous`, `alternatives`, and the depth fit
  (labels used/rejected, chunks OCR'd), legend, identical legend symbols, snapping grid, `warnings`
  and per-step `timings_s`.
* `IMAGE_overlay.png`: the whole logged range: the scan's depth and lithology columns side by side
  (squeezed vertically only, to at most 10,000 px), interval bars coloured by lithology, depth ticks
  from the result, and a colour key.
* `IMAGE_detail_1..3.png`: for logs too tall to show at full size, full-resolution 400 ft close-ups
  (top, middle, bottom) with each interval's depths and name.

## Pipeline (lithoextract/)

| Step | File | How |
|---|---|---|
| Depth calibration | `depth.py` | OCR the depth column (black ink only, so blue TVD labels are ignored; thousands separators like "9,120" are read). Label orientation (0/90/270) is decided once on probe chunks; very tall columns only have an evenly spaced sample of 12 chunks read (all chunks if that gives < 30 good labels). Readings off the dominant line (RANSAC over scales plausible for the DPI, 2 ft tolerance) or breaking increasing depth are thrown out; depth is interpolated between neighbouring labels. Warns if a stretch of labels follows a different scale part-way down, or if < 4 labels agree. |
| Interval boundaries | `features.py`, `segment.py` | Per-row stats (ink, white, fill hue, stroke lengths in ft), with grid lines erased; boundary where the window above and below differ; snap to the sharpest row; absorb slivers < 1 ft. The window is 2 ft, but never less than one vertical repeat of the column's symbols (measured by autocorrelation): on 1 in/100 ft logs a symbol repeat spans many feet, and a smaller window reads the symbol's own dash rows as bedding. |
| Grouping | `segment.py` | Length-weighted clustering of interval descriptors. |
| Legend | `legend.py` | Find drawn rectangles (with and without closing small gaps, so swatches stacked 1-2 px apart stay separate), including long thin bars (~0.6 x 0.08 in); a "bar" that is really a word fused by closing (covered by a readable rock name, paper between the letters) is dropped. Fill holes in the legend grid (a column of swatches meets a row of swatches, ink there, no box traced). OCR the text with every swatch blanked (pattern noise never glues onto a label; leading 1-2 letter junk is stripped; a line never spans a swatch), give each label the nearest box directly left of it, or above it when the legend puts labels under boxes; OCR fallback for unlabelled boxes, bounded by the next swatch or label. Symbols drawn with no box at all (chert triangles, a coal square, breccia clasts) are taken from the ink just left of a rock name. Words are snapped to a lithology vocabulary (rejoining OCR splits like "Anhyd rite"); ALL-CAPS names are title-cased. A legend with fewer than 2 symbols can't discriminate, so intervals are named `Unidentified (cluster k)`. |
| Group -> legend name | `shape.py`, `pipeline.py` | Scale-free stroke shape (share of ink in horizontal / vertical / diagonal strokes, dots, solid) + fill hue + ink density; nearest swatch; near-ties flagged `ambiguous`. "Sand and Sandstone" style names become the specific rock ("Sandstone"). Legend symbols drawn identically (ink match >= 0.9, e.g. Clay and Shale) resolve to the more common rock (Shale), flagged `ambiguous` with the other as an alternative. |
| Intervals | `snap.py`, `pipeline.py` | Blank stretches at either end dropped; neighbours with the same name merged. If most boundaries sit on a regular depth step (and that is very unlikely by chance) they are snapped to it. A piece shorter than one step with an off-grid edge has that edge moved to the nearest grid line if both sides survive (a bed whose faded top was detected low); otherwise it joins its most similar neighbour (a faded patch of the rock beside it). Pages without a step are left alone. |

Blank stretches at the top and bottom of the column are treated as "not logged" and dropped.
Blank stretches between intervals map to the legend's white swatch if there is one, with a cluster_id of their own.
Only the boxed regions are cropped and converted to RGB (scans stay in their native 1-bit mode), and the
image's size is checked against the annotation's `width`/`height`.

## Status on the two sample pages

| Page | Depth fit | Intervals | Notes |
|---|---|---|---|
| log06, Baker Hughes 400 DPI B&W | 16 labels, 20.03 px/ft | 17, all on the 10 ft grid | Shale/siltstone/sandstone split matches the scan. The 3 ft "sandstone" at 25497 (faded siltstone) is gone. **Clay and Shale legend symbols are identical**: reported as Shale, flagged ambiguous with Clay as the alternative. |
| Shell AC0336, Schlumberger 150 DPI colour | 625 labels from 12 of 48 chunks, 7.51 px/ft | 245 | No depth grid, so not snapped; same intervals as the full-OCR run, boundaries within 0.1 ft. ~5 s (was 29 s). |

Whole dataset (51 annotated images): 48 processed, ~4 min. Beds thinner than one symbol repeat on compressed logs are approximate or missed. 3 pages fail depth OCR (no labels read). The "Rock Types" legends (Rio Nueces/Lavaca/San Gabriel, Critter Creek, Callaway) now parse fully (23-31 symbols; previously 0-1, leaving every interval Unidentified).
Pages whose fit rests on < 4 labels are written but carry a warning in their JSON.

## Known gaps / next steps
1. Score against the 50 training pages (need their ground-truth format) and tune `thr`, `window_ft`, `merge_thr`.
2. Clay vs Shale (and any identical swatches): break ties with the description text beside the column
   ("Sh:", "Sltst:", "SHALE:") or a prior learned from training labels.
3. Steep diagonal hatching (Dolomite/Anhydrite) is not detected as diagonal yet; add an orientation
   histogram from image gradients.
4. Multi-line legend labels can lose a line ("Silty" instead of "Silty Sandstone").
5. Bonus: locate the three regions without boxes (OCR "LITHOLOGY"/"Depth" headers + pattern-texture search).
