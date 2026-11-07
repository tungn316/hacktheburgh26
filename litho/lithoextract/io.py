"""Load a log image and its bounding-box JSON.

The image is opened lazily and kept in its native mode (often 1-bit). Only the
boxed regions we actually look at are cropped and converted to RGB, so a
100,000-px-tall scan never has to exist at full size as RGB.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


@dataclass
class Page:
    path: Path
    image: Image.Image  # native mode, not converted
    dpi: float
    boxes: dict[str, tuple[int, int, int, int]]  # xyxy, x_max/y_max exclusive
    warnings: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.path.stem

    @property
    def width(self) -> int:
        return self.image.width

    @property
    def height(self) -> int:
        return self.image.height

    def crop_image(self, box: tuple[int, int, int, int]) -> Image.Image:
        return self.image.crop(box).convert("RGB")

    def crop(self, box: tuple[int, int, int, int]) -> np.ndarray:
        return np.asarray(self.crop_image(box))


def load_page(image_path: str | Path, boxes_path: str | Path | None = None) -> Page:
    image_path = Path(image_path)
    im = Image.open(image_path)
    dpi = float((im.info.get("dpi") or (0, 0))[0] or 0)
    boxes = {}
    if boxes_path:
        data = json.loads(Path(boxes_path).read_text())
        size = (data.get("width"), data.get("height"))
        if None not in size and tuple(size) != im.size:
            raise ValueError(f"annotation is for a {size[0]}x{size[1]} image but {image_path.name} is "
                             f"{im.width}x{im.height}: wrong JSON for this image?")
        boxes = {k: tuple(int(v) for v in b) for k, b in data["bounding_boxes"].items()}
        for k, (x0, y0, x1, y1) in boxes.items():
            if not (0 <= x0 < x1 <= im.width and 0 <= y0 < y1 <= im.height):
                raise ValueError(f"box {k}={[x0, y0, x1, y1]} does not fit inside the {im.width}x{im.height} image")
    return Page(image_path, im, dpi, boxes)


def ink_mask(rgb: np.ndarray, thresh: int = 110) -> np.ndarray:
    """Dark pixels (black/near-black ink): excludes coloured text such as blue TVD labels."""
    return (rgb.max(axis=2) < thresh)
