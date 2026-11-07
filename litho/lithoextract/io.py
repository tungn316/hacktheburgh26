"""Load a log image and its bounding-box JSON. Crops are taken lazily so huge scans stay cheap."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


@dataclass
class Page:
    path: Path
    image: Image.Image  # RGB
    dpi: float
    boxes: dict[str, tuple[int, int, int, int]]  # xyxy, x_max/y_max exclusive

    @property
    def name(self) -> str:
        return self.path.stem

    def crop(self, box: tuple[int, int, int, int]) -> np.ndarray:
        return np.asarray(self.image.crop(box))


def load_page(image_path: str | Path, boxes_path: str | Path | None = None) -> Page:
    image_path = Path(image_path)
    im = Image.open(image_path)
    dpi = float((im.info.get("dpi") or (0, 0))[0] or 0)
    im = im.convert("RGB")
    boxes = {}
    if boxes_path:
        data = json.loads(Path(boxes_path).read_text())
        boxes = {k: tuple(int(v) for v in b) for k, b in data["bounding_boxes"].items()}
    return Page(image_path, im, dpi, boxes)


def ink_mask(rgb: np.ndarray, thresh: int = 110) -> np.ndarray:
    """Dark pixels (black/near-black ink) — excludes coloured text such as blue TVD labels."""
    return (rgb.max(axis=2) < thresh)
