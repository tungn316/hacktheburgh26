"""Pictures for checking a result by eye.

* ``IMAGE_overlay.png``: the whole logged range. The scan's depth and lithology columns side by
  side, squeezed vertically only (so a 60 px wide column stays readable while a 100,000 px tall
  log fits in MAX_HEIGHT), the extracted intervals as bars coloured by lithology, depth ticks,
  and a colour key on top.
* ``IMAGE_detail_N.png``: for logs too tall to show at full size, up to three full-resolution
  close-ups (top, middle, bottom) with every interval's depths and name written beside it.

Only the boxed columns are cropped and converted to RGB, in strips, so tall 1-bit scans stay cheap.
"""
from __future__ import annotations

import colorsys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

MAX_HEIGHT = 10000  # overview height cap, px
DETAIL_SPAN_FT = 400
N_DETAILS = 3
STRIP = 4000  # rows converted to RGB at a time
BAR_W, TEXT_W, GAP = 50, 330, 12
PALETTE = [(40, 160, 60), (160, 100, 40), (240, 190, 0), (150, 60, 170), (0, 150, 150), (230, 110, 20),
           (200, 60, 120), (110, 110, 110), (0, 90, 200), (220, 40, 40)]


def _font(size: int):
    for name in ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def colour_map(names) -> dict[str, tuple[int, int, int]]:
    """A distinct colour per lithology, in order of first appearance (as in the example notebook)."""
    out: dict[str, tuple[int, int, int]] = {}
    for n in names:
        if n in out:
            continue
        i = len(out)
        if i < len(PALETTE):
            out[n] = PALETTE[i]
        else:
            h = (i * 0.618034) % 1.0
            out[n] = tuple(int(255 * c) for c in colorsys.hsv_to_rgb(h, 0.65, 0.8))
    return out


def _column_strip(page, box, y0: int, y1: int, height: int) -> Image.Image:
    """Crop columns box[0]:box[2] over rows y0:y1, resized to ``height`` rows, converting in strips."""
    x0, _, x1, _ = box
    out = Image.new("RGB", (x1 - x0, max(1, height)), "white")
    s = height / max(1, y1 - y0)
    for ys in range(y0, y1, STRIP):
        ye = min(ys + STRIP, y1)
        top, bot = round((ys - y0) * s), round((ye - y0) * s)
        if bot <= top:
            continue
        part = page.crop_image((x0, ys, x1, ye))
        if s != 1:
            part = part.resize((part.width, bot - top), Image.Resampling.BOX)
        out.paste(part, (0, top))
    return out


def _tick_step(px_per_ft_shown: float, min_px: float = 60) -> float:
    for step in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000):
        if step * px_per_ft_shown >= min_px:
            return step
    return 10000


def _render(page, result, y0: int, y1: int, scale: float, out: Path, colours, detailed: bool) -> None:
    ivs = result["intervals"]
    dbox, lbox = page.boxes["depth_column"], page.boxes["interpreted_lithology_column"]
    h = max(1, round((y1 - y0) * scale))
    dcol = _column_strip(page, dbox, y0, y1, h)
    lcol = _column_strip(page, lbox, y0, y1, h)
    font, small = _font(14), _font(12)

    # Colour key: only the lithologies present in this view.
    shown = [n for n in colours if any(iv["lithology"] == n and iv["bottom_y"] > y0 and iv["top_y"] < y1 for iv in ivs)]
    width = dcol.width + lcol.width + 2 * GAP + BAR_W + TEXT_W
    per_row = max(1, width // 220)
    key_h = 30 + 22 * ((len(shown) + per_row - 1) // per_row)
    canvas = Image.new("RGB", (width, h + key_h), "white")
    draw = ImageDraw.Draw(canvas)
    ppf = result["depth_fit"]["px_per_ft"]
    title = f"{result['image'][:60]}   " + (f"full column, vertical scale x{scale:.3f}" if not detailed else "full resolution")
    draw.text((6, 6), title, fill=(0, 0, 0), font=font)
    for k, n in enumerate(shown):
        x, y = 6 + (k % per_row) * 220, 28 + 22 * (k // per_row)
        draw.rectangle([x, y, x + 16, y + 14], fill=colours[n], outline=(0, 0, 0))
        draw.text((x + 22, y), n[:26], fill=(0, 0, 0), font=small)

    canvas.paste(dcol, (0, key_h))
    canvas.paste(lcol, (dcol.width + GAP, key_h))
    bx = dcol.width + lcol.width + 2 * GAP
    tx = bx + BAR_W + 6
    yy = lambda y: key_h + (y - y0) * scale  # noqa: E731
    for iv in ivs:
        if iv["bottom_y"] <= y0 or iv["top_y"] >= y1:
            continue
        a, b = yy(max(iv["top_y"], y0)), yy(min(iv["bottom_y"], y1))
        draw.rectangle([bx, a, bx + BAR_W, max(a, b - 1)], fill=colours[iv["lithology"]],
                       outline=(0, 0, 0) if b - a >= 3 else None)
        if detailed:
            draw.line([(dcol.width, a), (bx, a)], fill=(255, 0, 0), width=1)
            if b - a >= 14:
                label = f"{iv['top_depth']:g}-{iv['bottom_depth']:g}  {iv['lithology']}" + ("  (?)" if iv["ambiguous"] else "")
                draw.text((tx + 60, a + 1), label, fill=(0, 0, 0), font=small)
    # Depth ticks placed where the result itself puts each depth (interpolated within its interval),
    # so they can be compared with the scan's own depth labels on the left.
    step = _tick_step(ppf * scale)
    d = (int(ivs[0]["top_depth"] // step) + 1) * step
    i = 0
    while d < ivs[-1]["bottom_depth"]:
        while ivs[i]["bottom_depth"] < d:
            i += 1
        iv = ivs[i]
        f = (d - iv["top_depth"]) / (iv["bottom_depth"] - iv["top_depth"])
        y = iv["top_y"] + f * (iv["bottom_y"] - iv["top_y"])
        if y0 <= y < y1:
            draw.line([(tx, yy(y)), (tx + 10, yy(y))], fill=(0, 40, 200), width=2)
            draw.text((tx + 14, yy(y) - 7), f"{d:g}", fill=(0, 40, 200), font=small)
        d += step
    canvas.save(out, optimize=True)


def render_overlays(page, result: dict, out_dir: Path, stem: str) -> list[Path]:
    """Write the full-column overview and, for tall logs, full-resolution close-ups. Returns the paths."""
    ivs = result["intervals"]
    if not ivs:
        return []
    for old in out_dir.glob(f"{stem}_detail_*.png"):  # from an earlier run of the same image
        old.unlink()
    colours = colour_map(iv["lithology"] for iv in ivs)
    ppf = result["depth_fit"]["px_per_ft"]
    pad = int(5 * ppf)
    y0, y1 = max(0, ivs[0]["top_y"] - pad), min(page.height, ivs[-1]["bottom_y"] + pad)
    scale = min(1.0, MAX_HEIGHT / (y1 - y0))
    paths = [out_dir / f"{stem}_overlay.png"]
    _render(page, result, y0, y1, scale, paths[0], colours, detailed=scale == 1.0)
    span = int(DETAIL_SPAN_FT * ppf)
    if scale < 1.0 and y1 - y0 > span:
        starts = sorted({y0, (y0 + y1 - span) // 2, y1 - span})[:N_DETAILS]
        for k, s in enumerate(starts, 1):
            p = out_dir / f"{stem}_detail_{k}.png"
            _render(page, result, s, s + span, 1.0, p, colours, detailed=True)
            paths.append(p)
    return paths
