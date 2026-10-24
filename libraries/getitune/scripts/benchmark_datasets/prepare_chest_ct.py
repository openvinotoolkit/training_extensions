#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Download and prepare the chest CT benchmark dataset.

Uses a public archive with one image file per sample and exports it in the
experimental Datumaro format for multi-class classification benchmarks.

Archive: https://github.com/shreyabandyopadhyay/CTScanImageClassification/blob/main/DS3.zip
"""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path
from datumaro.experimental import Dataset, LazyImage
from datumaro.experimental.data_formats.coco.sample import CocoCategories, CocoSample
from datumaro.experimental.export_import import export_dataset
from datumaro.experimental.fields import ImageInfo, Subset
from PIL import Image

from getitune.benchmark.dataset_helpers import download, extract_archive, parse_args

_ARCHIVE_URL = "https://github.com/shreyabandyopadhyay/CTScanImageClassification/raw/main/DS3.zip"

_LABEL_NAMES = ("abdomen", "chest", "head")
_SPLIT_RATIOS = (0.7, 0.15, 0.15)


def _build_dataset(extracted_root: Path) -> Dataset:
    dataset: Dataset = Dataset(CocoSample, categories={"labels": CocoCategories(labels=_LABEL_NAMES)})
    image_id = 0
    class_files: dict[int, list[Path]] = {idx: [] for idx in range(len(_LABEL_NAMES))}
    for local_path in sorted(extracted_root.rglob("*.jpeg")):
        name = local_path.name.lower()
        if name.startswith("abdomen"):
            class_files[0].append(local_path)
        elif name.startswith("chest"):
            class_files[1].append(local_path)
        elif name.startswith("head"):
            class_files[2].append(local_path)

    for class_idx, files in class_files.items():
        if not files:
            continue
        total = len(files)
        train_end = int(total * _SPLIT_RATIOS[0])
        val_end = train_end + int(total * _SPLIT_RATIOS[1])
        split_map = (
            (Subset.TRAINING, files[:train_end]),
            (Subset.VALIDATION, files[train_end:val_end]),
            (Subset.TESTING, files[val_end:]),
        )
        for subset, subset_files in split_map:
            for local_path in subset_files:
                with Image.open(local_path) as im:
                    width, height = im.size
                dataset.append(
                    CocoSample(
                        image=LazyImage(local_path),
                        image_info=ImageInfo(width=width, height=height),
                        image_id=image_id,
                        subset=subset,
                        bboxes=None,
                        labels=np.asarray([class_idx], dtype=np.int64),
                        polygons=None,
                        areas=None,
                        iscrowd=None,
                        caption_group_ids=None,
                        captions=None,
                        keypoints=None,
                    ),
                )
                image_id += 1
    return dataset


def main() -> None:
    """Prepare the chest_ct benchmark dataset."""
    args = parse_args(description="Prepare the chest_ct benchmark dataset.")
    staging = args.archive_dir / f"{args.name}_raw"
    archive = download(_ARCHIVE_URL, dest_dir=args.archive_dir, filename=f"{args.name}.zip")
    extract_archive(archive, staging)
    dataset = _build_dataset(staging)
    if args.dest.exists():
        shutil.rmtree(args.dest)
    args.dest.parent.mkdir(parents=True, exist_ok=True)
    export_dataset(dataset, args.dest)
    archive.unlink(missing_ok=True)
    shutil.rmtree(staging, ignore_errors=True)


if __name__ == "__main__":
    main()
