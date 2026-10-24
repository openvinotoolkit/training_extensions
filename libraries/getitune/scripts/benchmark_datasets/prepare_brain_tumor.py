#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Download and prepare the brain tumor benchmark.

Uses the Kaggle dataset "medical-image-dataset-brain-tumor-segmentation" and
exports it in the experimental Datumaro format.

The Kaggle dataset ships (at least) two equivalent Roboflow YOLO-segmentation
exports (e.g. ``tumorSegmYolov9/``, ``tumorSegmentYolov8/``). Each export uses
the "subset-first" Roboflow layout::

    <export_root>/data.yaml
    <export_root>/train/images/*.jpg
    <export_root>/train/labels/*.txt
    <export_root>/valid/images/*.jpg
    <export_root>/valid/labels/*.txt
    <export_root>/test/images/*.jpg
    <export_root>/test/labels/*.txt

which is *not* the "images-first" Ultralytics convention
(``images/<subset>/`` + ``labels/<subset>/``) that Datumaro's generic YOLO
auto-import supports, and each label ``.txt`` line is a variable-length
polygon (``class_id x1 y1 x2 y2 ... xn yn``, normalized to ``[0, 1]``) rather
than a fixed-length bounding box — so this script parses the export directly
instead of delegating to a generic YOLO importer.

This dataset is gated behind a Kaggle account. Two ways to provide it:

1. Automatic download — requires benchmark dependencies installed via
    ``just venv-benchmark`` or ``uv sync --extra benchmark`` (from
    ``library/``), and API credentials
   configured via the ``KAGGLE_API_TOKEN`` environment variable, or a
   ``~/.kaggle/access_token`` file. See https://www.kaggle.com/docs/api.
2. Manual placement — download the dataset yourself (e.g. from
   https://www.kaggle.com/datasets/pkdarabi/medical-image-dataset-brain-tumor-segmentation)
   and pass ``--raw-dir <path>`` pointing at the extracted dataset root to skip
   the network step entirely while still running through this script's
   transform/export logic.
"""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING

import numpy as np
import yaml
from datumaro.experimental import Dataset, LazyImage
from datumaro.experimental.data_formats.coco.sample import CocoCategories, CocoSample
from datumaro.experimental.export_import import export_dataset
from datumaro.experimental.fields import ImageInfo, Subset
from PIL import Image

from getitune.benchmark.dataset_helpers import (
    download_kaggle_dataset,
    parse_args,
    resolve_raw_source,
)

if TYPE_CHECKING:
    from pathlib import Path

_KAGGLE_DATASET_ID = "pkdarabi/medical-image-dataset-brain-tumor-segmentation"

# Roboflow subset directory name -> Datumaro subset. Some exports use "val"
# instead of "valid"; both are accepted.
_SUBSET_DIR_TO_SUBSET = {
    "train": Subset.TRAINING,
    "valid": Subset.VALIDATION,
    "val": Subset.VALIDATION,
    "test": Subset.TESTING,
}


def _find_yolo_root(root: Path) -> Path:
    """Find a Roboflow-style ``data.yaml`` export root under *root*.

    Unlike the "images-first" Ultralytics convention, Roboflow exports are
    "subset-first": ``<root>/<subset>/images/`` + ``<root>/<subset>/labels/``
    for each of ``train`` / ``valid`` (or ``val``) / ``test``. This looks for
    a ``data.yaml`` whose sibling directories match that shape.
    """
    candidates = sorted(
        candidate
        for candidate in (path.parent for path in root.rglob("data.yaml"))
        if any((candidate / subset_dir / "images").is_dir() for subset_dir in _SUBSET_DIR_TO_SUBSET)
    )
    if not candidates:
        msg = f"No Roboflow-style YOLO dataset root found under {root}"
        raise RuntimeError(msg)
    return candidates[0]


def _load_label_names(yolo_root: Path) -> tuple[str, ...]:
    """Read class names (in class-id order) from a YOLO ``data.yaml``."""
    with (yolo_root / "data.yaml").open(encoding="utf-8") as f:
        data = yaml.safe_load(f)

    names = data.get("names", [])
    if isinstance(names, dict):
        return tuple(names[k] for k in sorted(names, key=int))
    return tuple(names)


def _polygon_area(coords: np.ndarray) -> float:
    """Shoelace-formula polygon area for absolute-pixel ``(N, 2)`` coordinates."""
    x, y = coords[:, 0], coords[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def _parse_yolo_segmentation_label(
    label_path: Path,
    width: int,
    height: int,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Parse a YOLO-segmentation label file into per-instance annotation arrays.

    Each line is ``class_id x1 y1 x2 y2 ... xn yn`` with coordinates
    normalized to ``[0, 1]``; converts to absolute-pixel ``xywh`` bboxes
    (axis-aligned bounding box of the polygon) and ``(N, 2)`` polygons, plus
    polygon-area and an all-``False`` ``iscrowd`` array.

    Returns ``(bboxes, labels, areas, iscrowd, polygons)``, all ``None`` when
    *label_path* doesn't exist or has no valid annotation lines.
    """
    if not label_path.exists():
        return None, None, None, None, None

    bboxes: list[list[float]] = []
    labels: list[int] = []
    polygons: list[np.ndarray] = []
    areas: list[float] = []

    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        parts = raw_line.split()
        # class_id + at least 3 (x, y) pairs -> odd token count >= 7.
        if len(parts) < 7 or len(parts) % 2 == 0:
            continue

        coords = np.asarray(parts[1:], dtype=np.float32).reshape(-1, 2)
        coords[:, 0] *= width
        coords[:, 1] *= height

        x_min, y_min = coords.min(axis=0)
        x_max, y_max = coords.max(axis=0)

        labels.append(int(parts[0]))
        bboxes.append([float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)])
        polygons.append(coords.astype(np.float32))
        areas.append(_polygon_area(coords))

    if not labels:
        return None, None, None, None, None

    polygons_arr = np.empty(len(polygons), dtype=object)
    for idx, poly in enumerate(polygons):
        polygons_arr[idx] = poly

    return (
        np.asarray(bboxes, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
        np.asarray(areas, dtype=np.float32),
        np.zeros(len(labels), dtype=np.bool_),
        polygons_arr,
    )


def _build_dataset(yolo_root: Path) -> Dataset:
    """Parse a Roboflow YOLO-segmentation export into a Datumaro ``CocoSample`` dataset."""
    dataset: Dataset = Dataset(
        CocoSample,
        categories={"labels": CocoCategories(labels=_load_label_names(yolo_root))},
    )

    image_id = 0
    for subset_dir_name, subset_enum in _SUBSET_DIR_TO_SUBSET.items():
        images_dir = yolo_root / subset_dir_name / "images"
        labels_dir = yolo_root / subset_dir_name / "labels"
        if not images_dir.is_dir():
            continue

        for image_path in sorted(p for p in images_dir.iterdir() if p.is_file()):
            with Image.open(image_path) as im:
                width, height = im.size

            bboxes, labels, areas, iscrowd, polygons = _parse_yolo_segmentation_label(
                labels_dir / f"{image_path.stem}.txt",
                width,
                height,
            )

            dataset.append(
                CocoSample(
                    image=LazyImage(image_path),
                    image_info=ImageInfo(width=width, height=height),
                    image_id=image_id,
                    subset=subset_enum,
                    bboxes=bboxes,
                    labels=labels,
                    polygons=polygons,
                    areas=areas,
                    iscrowd=iscrowd,
                    caption_group_ids=None,
                    captions=None,
                    keypoints=None,
                ),
            )
            image_id += 1

    if len(dataset) == 0:
        msg = f"No images found under YOLO dataset root {yolo_root}"
        raise RuntimeError(msg)

    return dataset


def main() -> None:
    """Prepare the brain_tumor dataset (Kaggle download or --raw-dir) and export it."""
    args = parse_args(description="Prepare the brain_tumor benchmark dataset.")

    def _download() -> Path:
        return download_kaggle_dataset(_KAGGLE_DATASET_ID)

    raw_root = resolve_raw_source(args, _download)
    dataset = _build_dataset(_find_yolo_root(raw_root))

    if args.dest.exists():
        shutil.rmtree(args.dest)
    args.dest.parent.mkdir(parents=True, exist_ok=True)
    export_dataset(dataset, args.dest)

    print(f"Dataset '{args.name}' ready at {args.dest}")


if __name__ == "__main__":
    main()
