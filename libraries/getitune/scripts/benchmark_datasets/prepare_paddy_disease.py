#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Download and prepare the paddy disease benchmark dataset.

Uses the public Hugging Face mirror of the Roboflow dataset and exports it in
the experimental Datumaro format for multi-class classification benchmarks.

Source: https://universe.roboflow.com/paddy-kkdef/paddy-bc4ue
Mirror: https://huggingface.co/datasets/anthony2261/paddy-disease-classification
"""

from __future__ import annotations

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

_REPO = "anthony2261/paddy-disease-classification"
_REVISION = "774edfc6bf92e9ca0ad29aeb5c1eccf4e9968182"
_SPLIT_FILES = {
    "train": "data/train-00000-of-00002-a929229f3aaf7166.parquet",
    "validation": "data/train-00001-of-00002-4c2a20b9469e90df.parquet",
}

_LABEL_NAMES = (
    "bacterial_leaf_blight",
    "bacterial_leaf_streak",
    "bacterial_panicle_blight",
    "blast",
    "brown_spot",
    "dead_heart",
    "downy_mildew",
    "hispa",
    "normal",
    "tungro",
)


def _stratified_split(
    examples: list[tuple[dict, int]],
    *,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[tuple[dict, int]], list[tuple[dict, int]], list[tuple[dict, int]]]:
    """Deterministically split examples into train/val/test stratified by label."""
    rng = np.random.default_rng(seed)
    per_label: dict[int, list[tuple[dict, int]]] = {}
    for example in examples:
        per_label.setdefault(example[1], []).append(example)

    train_examples: list[tuple[dict, int]] = []
    validation_examples: list[tuple[dict, int]] = []
    testing_examples: list[tuple[dict, int]] = []
    for label in sorted(per_label):
        group = per_label[label]
        rng.shuffle(group)
        total = len(group)
        train_end = round(total * train_ratio)
        val_end = train_end + round(total * val_ratio)
        train_examples.extend(group[:train_end])
        validation_examples.extend(group[train_end:val_end])
        testing_examples.extend(group[val_end:])
    return train_examples, validation_examples, testing_examples


def _build_dataset(parquet_paths: dict[Subset, Path], images_dir: Path) -> Dataset:
    images_dir.mkdir(parents=True, exist_ok=True)
    dataset: Dataset = Dataset(CocoSample, categories={"labels": CocoCategories(labels=_LABEL_NAMES)})

    examples: list[tuple[dict, int]] = []
    for parquet_path in parquet_paths.values():
        parquet_file = pq.ParquetFile(str(parquet_path))
        for batch in parquet_file.iter_batches(batch_size=256):
            images = batch.column("image").to_pylist()
            labels = batch.column("label").to_pylist()
            examples.extend(zip(images, (int(label) for label in labels), strict=True))

    if not examples:
        return dataset

    train_examples, validation_examples, testing_examples = _stratified_split(examples)

    image_id = 0
    for subset, subset_examples in (
        (Subset.TRAINING, train_examples),
        (Subset.VALIDATION, validation_examples),
        (Subset.TESTING, testing_examples),
    ):
        for image_struct, label in subset_examples:
            img_bytes = image_struct["bytes"]
            suffix = image_struct.get("path", "image.jpg").rsplit(".", 1)[-1].lower()
            img_path = images_dir / f"paddy_{image_id:05d}.{suffix}"
            img_path.write_bytes(img_bytes)
            with Image.open(img_path) as im:
                width, height = im.size
            dataset.append(
                CocoSample(
                    image=LazyImage(img_path),
                    image_info=ImageInfo(width=width, height=height),
                    image_id=image_id,
                    subset=subset,
                    bboxes=None,
                    labels=np.asarray([label], dtype=np.int64),
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
    """Prepare the paddy_disease benchmark dataset."""
    args = parse_args(description="Prepare the paddy_disease benchmark dataset.")
    parquet_paths: dict[Subset, Path] = {}
    for split, path in _SPLIT_FILES.items():
        parquet_paths[Subset.TRAINING if split == "train" else Subset.VALIDATION] = download(
            f"https://huggingface.co/datasets/{_REPO}/resolve/{_REVISION}/{path}",
            dest_dir=args.archive_dir,
            filename=f"{args.name}_{split}.parquet",
        )

    staging = args.archive_dir / f"{args.name}_raw"
    dataset = _build_dataset(parquet_paths, staging / "images")
    if args.dest.exists():
        shutil.rmtree(args.dest)
    args.dest.parent.mkdir(parents=True, exist_ok=True)
    export_dataset(dataset, args.dest)
    for parquet_path in parquet_paths.values():
        parquet_path.unlink(missing_ok=True)
    shutil.rmtree(staging, ignore_errors=True)


if __name__ == "__main__":
    main()
