#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Download and prepare the Oxford-IIIT Pet benchmark dataset.

Downloads the Oxford-IIIT Pet dataset (~3.7k images, 37 pet categories with
pixel-level trimap segmentation masks). Exports in the experimental Datumaro
dataset format for semantic segmentation benchmarks.

Source: https://www.robots.ox.ac.uk/~vgg/data/pets/
"""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING

import numpy as np
from datumaro.experimental import Dataset, LazyImage
from datumaro.experimental.data_formats.coco.sample import CocoCategories, CocoSample
from datumaro.experimental.export_import import export_dataset
from datumaro.experimental.fields import ImageInfo, Subset
from PIL import Image

from getitune.benchmark.dataset_helpers import download, extract_archive, parse_args

if TYPE_CHECKING:
    from pathlib import Path

_IMAGES_URL = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
_ANNOTATIONS_URL = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"

# Trimap values: 1=foreground, 2=background, 3=border
# We map to: 0=background, 1=foreground, ignore border (treat as background)
_LABEL_NAMES = ("background", "pet")


def _load_trimap_as_mask(trimap_path: Path) -> np.ndarray:
    """Load a trimap PNG and convert to binary mask (0=background, 1=foreground)."""
    trimap = np.array(Image.open(trimap_path))
    # 1 = foreground in the trimap convention
    return (trimap == 1).astype(np.uint8)


def _mask_to_polygon(mask: np.ndarray) -> list[np.ndarray]:
    """Convert a binary mask to a list of polygon arrays for CocoSample.

    Returns a simple bounding-box polygon derived from the mask's bounding rect.
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return []
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    # Rectangle polygon as (N, 2) array
    poly = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]], dtype=np.float32)
    return [poly]


def _build_dataset(staging: Path) -> Dataset:
    """Parse Oxford-IIIT Pet and build a Datumaro CocoSample dataset."""
    images_dir = staging / "images"
    trimaps_dir = staging / "annotations" / "trimaps"
    splits_dir = staging / "annotations"

    dataset: Dataset = Dataset(
        CocoSample,
        categories={"labels": CocoCategories(labels=_LABEL_NAMES)},
    )

    # Read official splits
    def _read_split(filename: str) -> dict[str, str]:
        """Return {image_name: class_id} from a split file."""
        result = {}
        split_file = splits_dir / filename
        if not split_file.exists():
            return result
        for line in split_file.read_text().splitlines():
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split()
            result[parts[0]] = parts[1]
        return result

    trainval_images = _read_split("trainval.txt")
    test_images = _read_split("test.txt")

    # Deterministic val split: every 5th trainval image
    trainval_sorted = sorted(trainval_images.keys())
    val_set = set(trainval_sorted[::5])

    all_images = {}
    for name in trainval_images:
        if name in val_set:
            all_images[name] = Subset.VALIDATION
        else:
            all_images[name] = Subset.TRAINING
    for name in test_images:
        all_images[name] = Subset.TESTING

    for img_idx, (img_name, subset) in enumerate(sorted(all_images.items())):
        img_path = images_dir / f"{img_name}.jpg"
        trimap_path = trimaps_dir / f"{img_name}.png"

        if not img_path.exists() or not trimap_path.exists():
            continue

        mask = _load_trimap_as_mask(trimap_path)
        height, width = mask.shape

        # Convert mask to bbox + polygon for CocoSample
        polygons = _mask_to_polygon(mask)
        if polygons:
            poly = polygons[0]
            x_min, y_min = poly[0]
            x_max, y_max = poly[2]
            bbox = np.array([[x_min, y_min, x_max - x_min, y_max - y_min]], dtype=np.float32)
            labels = np.array([1], dtype=np.int64)  # "pet" class
            areas = bbox[:, 2] * bbox[:, 3]
            poly_arr = np.empty(1, dtype=object)
            poly_arr[0] = poly
        else:
            bbox = None
            labels = None
            areas = None
            poly_arr = None

        dataset.append(
            CocoSample(
                image=LazyImage(img_path),
                image_info=ImageInfo(width=width, height=height),
                image_id=img_idx,
                subset=subset,
                bboxes=bbox,
                labels=labels,
                polygons=poly_arr,
                areas=areas,
                iscrowd=None,
                caption_group_ids=None,
                captions=None,
                keypoints=None,
            ),
        )

    return dataset


def main() -> None:
    """Download Oxford-IIIT Pet, convert to Datumaro format, and save."""
    args = parse_args(description="Prepare the oxford_pets benchmark dataset.")

    # Download images and annotations
    images_archive = download(_IMAGES_URL, dest_dir=args.archive_dir, filename=f"{args.name}_images.tar.gz")
    ann_archive = download(_ANNOTATIONS_URL, dest_dir=args.archive_dir, filename=f"{args.name}_annotations.tar.gz")

    # Extract into staging
    staging = args.archive_dir / f"{args.name}_raw"
    extract_archive(images_archive, staging, clean_dest=True)
    extract_archive(ann_archive, staging, clean_dest=False)
    images_archive.unlink(missing_ok=True)
    ann_archive.unlink(missing_ok=True)

    print("Building Datumaro dataset from Oxford-IIIT Pet ...")
    dataset = _build_dataset(staging)
    print(f"  Dataset length: {len(dataset)}")

    if args.dest.exists():
        shutil.rmtree(args.dest)
    args.dest.parent.mkdir(parents=True, exist_ok=True)

    print(f"Exporting dataset to {args.dest} ...")
    export_dataset(dataset, args.dest)

    # Cleanup
    shutil.rmtree(staging, ignore_errors=True)

    print(f"Dataset '{args.name}' ready at {args.dest}")


if __name__ == "__main__":
    main()
