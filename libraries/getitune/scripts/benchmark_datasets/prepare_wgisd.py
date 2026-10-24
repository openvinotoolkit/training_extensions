#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Download and prepare the 'wgisd' benchmark dataset.

Downloads the WGISD (Wine Grape Instance Segmentation Dataset) from GitHub,
parses its COCO-style polygon annotations, and exports the result in the
experimental Datumaro dataset format.
"""

from __future__ import annotations

import json
import shutil
from typing import TYPE_CHECKING

import numpy as np
from datumaro.experimental import Dataset, LazyImage
from datumaro.experimental.data_formats.coco.sample import CocoCategories, CocoSample
from datumaro.experimental.export_import import export_dataset
from datumaro.experimental.fields import ImageInfo, Subset

from getitune.benchmark.dataset_helpers import download, extract_archive, parse_args

if TYPE_CHECKING:
    from pathlib import Path

_COMMIT = "6910edc5ae3aae8c20062941b1641821f0c30127"
_URL = f"https://github.com/thsant/wgisd/archive/{_COMMIT}.zip"

# WGISD ships two COCO-style polygon annotation files inside ``coco_annotations/``.
# It does not provide a dedicated validation split, so we deterministically carve
# one out of the training images below (see ``_VAL_FRACTION``).
_TRAIN_FILE = "train_polygons_instances.json"
_TEST_FILE = "test_polygons_instances.json"

# Fraction of training images reassigned to the VALIDATION subset.
# The split is deterministic: images are sorted by ``id`` and every Nth image
# (where ``N = round(1 / _VAL_FRACTION)``) is moved to validation. Running the
# script multiple times therefore always produces the exact same subset
# assignment.
_VAL_FRACTION = 0.2


def _polygon_to_array(segmentation: list[list[float]]) -> np.ndarray:
    """Convert a COCO polygon (list of flat ``[x, y, x, y, ...]`` rings) to ``(N, 2)`` float32."""
    # Use the first (outer) ring; WGISD annotations have a single ring per instance.
    return np.asarray(segmentation[0], dtype=np.float32).reshape(-1, 2)


def _build_dataset(extracted_root: Path) -> Dataset:
    """Parse WGISD COCO annotations and build a Datumaro ``CocoSample`` dataset.

    The training file is split deterministically into TRAINING / VALIDATION
    subsets (see ``_VAL_FRACTION``); the test file is mapped to TESTING.
    """
    ann_dir = extracted_root / "coco_annotations"
    images_dir = extracted_root / "data"

    # Categories are identical across train/test files; read them from the train file.
    with (ann_dir / _TRAIN_FILE).open() as f:
        train_ann = json.load(f)

    categories_sorted = sorted(train_ann["categories"], key=lambda c: c["id"])
    label_names = tuple(c["name"] for c in categories_sorted)
    cat_id_to_idx = {c["id"]: idx for idx, c in enumerate(categories_sorted)}

    dataset: Dataset = Dataset(
        CocoSample,
        categories={"labels": CocoCategories(labels=label_names)},
    )

    # ----- training file → TRAINING + VALIDATION (deterministic split) -----
    train_images = sorted(train_ann["images"], key=lambda im: im["id"])
    val_step = max(1, round(1 / _VAL_FRACTION))
    # Every ``val_step``-th image (starting at index 0) becomes validation.
    val_indices = set(range(0, len(train_images), val_step))

    train_anns_by_img: dict[int, list[dict]] = {}
    for ann in train_ann["annotations"]:
        train_anns_by_img.setdefault(ann["image_id"], []).append(ann)

    for idx, img in enumerate(train_images):
        subset = Subset.VALIDATION if idx in val_indices else Subset.TRAINING
        _append_image(
            dataset,
            img,
            train_anns_by_img.get(img["id"], []),
            images_dir,
            subset,
            cat_id_to_idx,
        )

    # ----- test file → TESTING -----
    with (ann_dir / _TEST_FILE).open() as f:
        test_ann = json.load(f)

    test_anns_by_img: dict[int, list[dict]] = {}
    for ann in test_ann["annotations"]:
        test_anns_by_img.setdefault(ann["image_id"], []).append(ann)

    for img in sorted(test_ann["images"], key=lambda im: im["id"]):
        _append_image(
            dataset,
            img,
            test_anns_by_img.get(img["id"], []),
            images_dir,
            Subset.TESTING,
            cat_id_to_idx,
        )

    return dataset


def _append_image(
    dataset: Dataset,
    img: dict,
    anns: list[dict],
    images_dir: Path,
    subset: Subset,
    cat_id_to_idx: dict[int, int],
) -> None:
    """Build a single ``CocoSample`` and append it to *dataset*."""
    img_id = img["id"]
    width, height = int(img["width"]), int(img["height"])

    if anns:
        bboxes = np.asarray([a["bbox"] for a in anns], dtype=np.float32)
        labels = np.asarray([cat_id_to_idx[a["category_id"]] for a in anns], dtype=np.int64)
        areas = np.asarray([a.get("area", 0.0) for a in anns], dtype=np.float32)
        iscrowd = np.asarray([a.get("iscrowd", 0) for a in anns], dtype=np.int32)
        polygons = np.asarray(
            [_polygon_to_array(a["segmentation"]) for a in anns],
            dtype=object,
        )
    else:
        bboxes = None
        labels = None
        areas = None
        iscrowd = None
        polygons = None

    dataset.append(
        CocoSample(
            image=LazyImage(images_dir / img["file_name"]),
            image_info=ImageInfo(width=width, height=height),
            image_id=img_id,
            subset=subset,
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


def main() -> None:
    """Download WGISD, convert it to the experimental Datumaro format, and save it."""
    args = parse_args(description="Prepare the wgisd benchmark dataset.")

    archive = download(_URL, dest_dir=args.archive_dir, filename=f"{args.name}.zip")

    # Extract into a temporary staging directory; ``args.dest`` will hold the
    # final Datumaro dataset.
    staging = args.archive_dir / f"{args.name}_raw"
    extract_archive(archive, staging)
    archive.unlink(missing_ok=True)

    # The GitHub archive extracts into a single top-level ``wgisd-<commit>/`` folder.
    extracted_root = next(p for p in staging.iterdir() if p.is_dir())

    print("Building Datumaro dataset from WGISD COCO annotations ...")
    dataset = _build_dataset(extracted_root)
    print(f"  Dataset length: {len(dataset)}")

    # ``export_dataset`` requires that the output path does NOT exist yet,
    # so remove any leftover from a previous run and let it create the dir.
    if args.dest.exists():
        shutil.rmtree(args.dest)
    args.dest.parent.mkdir(parents=True, exist_ok=True)

    print(f"Exporting dataset to {args.dest} ...")
    export_dataset(dataset, args.dest)

    # Clean up extracted archive contents.
    shutil.rmtree(staging, ignore_errors=True)

    print(f"Dataset '{args.name}' ready at {args.dest}")


if __name__ == "__main__":
    main()
