#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Download and prepare the Oxford Flowers 102 benchmark dataset.

Downloads the Oxford Flowers 102 dataset (102 flower categories, ~8k images)
and exports it in the experimental Datumaro dataset format for multi-class
classification benchmarks.

Source: https://www.robots.ox.ac.uk/~vgg/data/flowers/102/
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
from scipy.io import loadmat

from getitune.benchmark.dataset_helpers import download, extract_archive, parse_args

if TYPE_CHECKING:
    from pathlib import Path

_IMAGES_URL = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
_LABELS_URL = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat"
_SPLITS_URL = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat"

_NUM_CLASSES = 102


def _build_dataset(extracted_root: Path, labels_path: Path, splits_path: Path) -> Dataset:
    """Parse Flowers 102 and build a Datumaro CocoSample dataset."""
    images_dir = extracted_root / "jpg"

    # Labels: 1-based array of shape (1, 8189)
    labels_mat = loadmat(str(labels_path))
    all_labels = labels_mat["labels"].flatten() - 1  # convert to 0-based

    # Splits: trnid, valid, tstid — each a 1-based array of image indices
    splits_mat = loadmat(str(splits_path))
    train_ids = set(splits_mat["trnid"].flatten().tolist())
    val_ids = set(splits_mat["valid"].flatten().tolist())

    label_names = tuple(f"flower_{i:03d}" for i in range(_NUM_CLASSES))

    dataset: Dataset = Dataset(
        CocoSample,
        categories={"labels": CocoCategories(labels=label_names)},
    )

    num_images = len(all_labels)
    for img_idx in range(1, num_images + 1):  # 1-based image numbering
        filename = f"image_{img_idx:05d}.jpg"
        img_path = images_dir / filename

        if img_idx in train_ids:
            subset = Subset.TRAINING
        elif img_idx in val_ids:
            subset = Subset.VALIDATION
        else:
            subset = Subset.TESTING

        label = int(all_labels[img_idx - 1])

        with Image.open(img_path) as im:
            width, height = im.size

        dataset.append(
            CocoSample(
                image=LazyImage(img_path),
                image_info=ImageInfo(width=width, height=height),
                image_id=img_idx,
                subset=subset,
                bboxes=None,
                labels=np.array([label], dtype=np.int64),
                polygons=None,
                areas=None,
                iscrowd=None,
                caption_group_ids=None,
                captions=None,
                keypoints=None,
            ),
        )

    return dataset


def main() -> None:
    """Download Flowers 102, convert to Datumaro format, and save."""
    args = parse_args(description="Prepare the flowers102 benchmark dataset.")

    # Download all components
    images_archive = download(_IMAGES_URL, dest_dir=args.archive_dir, filename=f"{args.name}_images.tgz")
    labels_path = download(_LABELS_URL, dest_dir=args.archive_dir, filename=f"{args.name}_labels.mat")
    splits_path = download(_SPLITS_URL, dest_dir=args.archive_dir, filename=f"{args.name}_splits.mat")

    # Extract images
    staging = args.archive_dir / f"{args.name}_raw"
    extract_archive(images_archive, staging)
    images_archive.unlink(missing_ok=True)

    print("Building Datumaro dataset from Flowers 102 ...")
    dataset = _build_dataset(staging, labels_path, splits_path)
    print(f"  Dataset length: {len(dataset)}")

    if args.dest.exists():
        shutil.rmtree(args.dest)
    args.dest.parent.mkdir(parents=True, exist_ok=True)

    print(f"Exporting dataset to {args.dest} ...")
    export_dataset(dataset, args.dest)

    # Cleanup
    shutil.rmtree(staging, ignore_errors=True)
    labels_path.unlink(missing_ok=True)
    splits_path.unlink(missing_ok=True)

    print(f"Dataset '{args.name}' ready at {args.dest}")


if __name__ == "__main__":
    main()
