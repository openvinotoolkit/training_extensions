# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Synthetic COCO instance-segmentation data for the Phase 0 parameter sweep."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class SweepCell:
    """One point in the image-size x instances x vertices sweep."""

    image_size: int
    instances: int
    vertices: int
    num_images: int = 32

    @property
    def slug(self) -> str:
        """Filesystem-safe identifier for this cell."""
        return f"s{self.image_size}_n{self.instances}_v{self.vertices}"


#: Diagonal first. The full 3x4x3 cross-product is 36 cells; expand only where
#: the curve bends. The *vertices* axis is the one that separates rasterization
#: cost (roughly flat in vertex count) from polygon-transform cost (linear in
#: it) -- that distinction is what justifies Phase 2 over Phase 5.
DIAGONAL_SWEEP: tuple[SweepCell, ...] = (
    SweepCell(512, 1, 4),
    SweepCell(1024, 10, 20),
    SweepCell(2048, 50, 100),
)

FULL_SWEEP: tuple[SweepCell, ...] = tuple(
    SweepCell(size, n, v) for size in (512, 1024, 2048) for n in (1, 10, 50, 200) for v in (4, 20, 100)
)


def _blob_polygon(cx: float, cy: float, radius: float, vertices: int, rng: np.random.Generator) -> list[float]:
    """Generate a closed star-shaped polygon as a flat ``[x0, y0, x1, y1, ...]`` list."""
    angles = np.linspace(0.0, 2.0 * np.pi, vertices, endpoint=False)
    radii = radius * rng.uniform(0.6, 1.0, size=vertices)
    xs = cx + radii * np.cos(angles)
    ys = cy + radii * np.sin(angles)
    return np.stack([xs, ys], axis=1).ravel().round(2).tolist()


def make_synthetic_coco(root: Path, cell: SweepCell, seed: int = 0) -> Path:
    """Write a COCO instance-segmentation dataset for one sweep cell.

    Args:
        root: Directory to create the dataset in.
        cell: Sweep parameters.
        seed: RNG seed for polygon placement and image content.

    Returns:
        The dataset root, laid out as ``<root>/images/train/`` plus
        ``<root>/annotations/instances_train.json``.
    """
    rng = np.random.default_rng(seed)
    img_dir = root / "images" / "train"
    ann_dir = root / "annotations"
    img_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    size = cell.image_size
    images: list[dict] = []
    annotations: list[dict] = []
    ann_id = 1

    for img_id in range(cell.num_images):
        # Structured noise, not flat colour: flat images can let JPEG decode and
        # downstream resize hit unrepresentative fast paths.
        pixels = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
        file_name = f"{img_id:06d}.jpg"
        Image.fromarray(pixels).save(img_dir / file_name, quality=90)
        images.append({"id": img_id, "file_name": file_name, "height": size, "width": size})

        radius = size / (2.0 * max(1, int(np.sqrt(cell.instances))) + 2.0)
        for _ in range(cell.instances):
            cx = float(rng.uniform(radius, size - radius))
            cy = float(rng.uniform(radius, size - radius))
            poly = _blob_polygon(cx, cy, radius, cell.vertices, rng)
            xs, ys = poly[0::2], poly[1::2]
            x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 1,
                    "segmentation": [poly],
                    "bbox": [x0, y0, x1 - x0, y1 - y0],
                    "area": float((x1 - x0) * (y1 - y0)),
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    payload = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "blob", "supercategory": "blob"}],
    }
    (ann_dir / "instances_train.json").write_text(json.dumps(payload))
    return root
