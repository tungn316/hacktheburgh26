2 lithoextract: lithology intervals from scanned well logs

```
pip install -r requirements.txt            # + tesseract binary
python run.py IMAGE.tif BOXES.json --out results/
python run.py --dir data/ --out results/   # every IMAGE.tif with a matching IMAGE.json
```

Outputs per image: `IMAGE.json` (everything), `IMAGE.csv` (top_ft, base_ft, group, lithology,
confidence, ambiguous, alternatives), `IMAGE_overlay.png` (scan beside the extracted intervals).

## Pipeline (lithoextract/)

| Step | File | How |
|---|---|---|
| Depth calibration | `depth.py` | OCR the depth column (black ink only, so blue TVD labels are ignored), trying 0/90/270 degree rotation; robust straight-line fit pixel row -> ft. |
| Interval boundaries | `features.py`, `segment.py` | Per-row stats (ink, white, fill hue, stroke lengths in ft), with grid lines erased; boundary where the window above and below differ; snap to the sharpest row; absorb slivers < 1 ft. |
| Grouping | `segment.py` | Length-weighted clustering of interval descriptors. |
| Legend | `legend.py` | Find drawn rectangles, OCR the text, give each label the largest box directly left of or above it; OCR fallback for unlabelled boxes; snap words to a lithology vocabulary. |
| Group -> legend name | `shape.py`, `pipeline.py` | Scale-free stroke shape (share of ink in horizontal / vertical / diagonal strokes, dots, solid) + fill hue + ink density; nearest swatch; near-ties flagged `ambiguous`. |

Blank stretches at the top and bottom of the column are treated as "not logged" and dropped.
Blank stretches between intervals map to the legend's white swatch if there is one.

## Status on the two sample pages

| Page | Depth fit | Intervals | Notes |
|---|---|---|---|
| Baker Hughes 400 DPI B&W | 16/17 labels, 20.03 px/ft, 0.04 ft mean error | 18 | Shale/siltstone/sand split correctly; **Clay vs Shale legend patterns are identical**, so flagged ambiguous |
| Schlumberger 150 DPI colour | 1879 labels, 7.51 px/ft, 0.03 ft | 245 | Colour fills make matching easy; 43 s, mostly OCR |

## Known gaps / next steps
1. Score against the 50 training pages (need their ground-truth format) and tune `thr`, `window_ft`, `merge_thr`.
2. Clay vs Shale (and any identical swatches): break ties with the description text beside the column
   ("Sh:", "Sltst:", "SHALE:") or a prior learned from training labels.
3. Steep diagonal hatching (Dolomite/Anhydrite) is not detected as diagonal yet; add an orientation
   histogram from image gradients.
4. Multi-line legend labels can lose a line ("Silty" instead of "Silty Sandstone").
5. Bonus: locate the three regions without boxes (OCR "LITHOLOGY"/"Depth" headers + pattern-texture search).
6. Speed: OCR every 2nd chunk of very tall depth columns; the fit only needs a few dozen labels.
