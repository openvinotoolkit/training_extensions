#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Download and prepare the 'aid_multilabel' benchmark dataset.

Downloads the AID Multi-Label aerial-scene dataset (3,000 overhead images, 17
co-occurring land-cover labels) from its public HuggingFace mirror and exports
it as an image-level **multi-label classification** dataset in the experimental
Datumaro format.

This is a privacy-safe replacement for VOC2007 multi-label: AID consists solely
of overhead aerial imagery (CC0 / public domain) with no people or other
privacy-sensitive content.

The upstream dataset ships a single ``train`` split, so train / validation /
test subsets are carved out deterministically (see :data:`_SUBSET_PERIOD`).

Source: https://huggingface.co/datasets/jonathan-roberts1/AID_MultiLabel
Imagery: Xia et al., "AID: A benchmark data set for performance evaluation of
aerial scene classification", IEEE TGRS 2017.
Multi-labels: Hua et al., "Relation Network for Multi-label Aerial Image
Classification", IEEE TGRS 2019.
"""

from __future__ import annotations

import io
import shutil
from typing import TYPE_CHECKING

import numpy as np
import pyarrow.parquet as pq
from datumaro.experimental import Dataset, LazyImage
from datumaro.experimental.data_formats.coco.sample import CocoCategories, CocoSample
from datumaro.experimental.export_import import export_dataset
from datumaro.experimental.fields import ImageInfo, Subset
from PIL import Image

from getitune.benchmark.dataset_helpers import download, parse_args

if TYPE_CHECKING:
    from pathlib import Path

# Pinned revision of the HuggingFace dataset for reproducibility.
_REVISION = "a62162cc5d869af69bf3d84b1bd2db898c0f6b5d"
_PARQUET_URL = (
    f"https://huggingface.co/datasets/jonathan-roberts1/AID_MultiLabel/resolve/{_REVISION}/"
    "data/train-00000-of-00001-ee58cb5d786e111e.parquet"
)

# The 17 multi-label classes, in the index order defined by the upstream dataset.
_LABEL_NAMES = (
    "airplane",
    "bare soil",
    "buildings",
    "cars",
    "chaparral",
    "court",
    "dock",
    "field",
    "grass",
    "mobile home",
    "pavement",
    "sand",
    "sea",
    "ship",
    "tanks",
    "trees",
    "water",
)

# Deterministic subset assignment. Within every window of ``_SUBSET_PERIOD``
# samples (ordered by row index), the first few go to TESTING, the next few to
# VALIDATION, and the remainder to TRAINING. This yields a reproducible
# ~70 / 15 / 15 train/val/test split (2100 / 450 / 450 for 3000 images).
_SUBSET_PERIOD = 20
_TEST_CUTOFF = 3  # indices [0, 3)   → TESTING
_VAL_CUTOFF = 6  # indices [3, 6)   → VALIDATION  (remainder → TRAINING)


def _subset_for_index(idx: int) -> Subset:
    """Return the deterministic subset for the sample at position ``idx``."""
    position = idx % _SUBSET_PERIOD
    if position < _TEST_CUTOFF:
        return Subset.TESTING
    if position < _VAL_CUTOFF:
        return Subset.VALIDATION
    return Subset.TRAINING


def _build_dataset(parquet_path: Path, images_dir: Path) -> Dataset:
    """Materialise images from the parquet and build a multi-label ``CocoSample`` dataset."""
    images_dir.mkdir(parents=True, exist_ok=True)

    dataset: Dataset = Dataset(
        CocoSample,
        categories={"labels": CocoCategories(labels=_LABEL_NAMES)},
    )

    parquet_file = pq.ParquetFile(str(parquet_path))
    image_id = 0
    for batch in parquet_file.iter_batches(batch_size=256):
        images = batch.column("image").to_pylist()
        labels = batch.column("label").to_pylist()
        for image_struct, label_indices in zip(images, labels, strict=True):
            img_bytes = image_struct["bytes"]

            with Image.open(io.BytesIO(img_bytes)) as im:
                width, height = im.size

            # Persist the embedded JPEG so the exporter can copy it into the
            # final dataset directory.
            img_path = images_dir / f"aid_{image_id:05d}.jpg"
            img_path.write_bytes(img_bytes)

            dataset.append(
                CocoSample(
                    image=LazyImage(img_path),
                    image_info=ImageInfo(width=width, height=height),
                    image_id=image_id,
                    subset=_subset_for_index(image_id),
                    bboxes=None,
                    labels=np.asarray(sorted(label_indices), dtype=np.int64),
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
    """Download AID Multi-Label, convert it to a multi-label Datumaro dataset, and save it."""
    args = parse_args(description="Prepare the aid_multilabel benchmark dataset.")

    parquet_path = download(_PARQUET_URL, dest_dir=args.archive_dir, filename=f"{args.name}.parquet")

    # Staging directory for the JPEGs materialised from the parquet; ``args.dest``
    # will hold the final Datumaro dataset.
    staging = args.archive_dir / f"{args.name}_raw"
    images_dir = staging / "images"

    print("Building multi-label Datumaro dataset from AID Multi-Label ...")
    dataset = _build_dataset(parquet_path, images_dir)
    print(f"  Dataset length: {len(dataset)}")

    # ``export_dataset`` requires that the output path does NOT exist yet,
    # so remove any leftover from a previous run and let it create the dir.
    if args.dest.exists():
        shutil.rmtree(args.dest)
    args.dest.parent.mkdir(parents=True, exist_ok=True)

    print(f"Exporting dataset to {args.dest} ...")
    export_dataset(dataset, args.dest)

    # Clean up the parquet and staged images.
    parquet_path.unlink(missing_ok=True)
    shutil.rmtree(staging, ignore_errors=True)

    print(f"Dataset '{args.name}' ready at {args.dest}")


if __name__ == "__main__":
    main()
