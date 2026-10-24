#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Download and prepare the Food-101 benchmark dataset.

Uses the Kaggle mirror of Food-101 and exports it in the experimental
Datumaro format for multi-class classification benchmarks.

Source: https://www.kaggle.com/datasets/kmader/food41

The archive follows the standard Food-101 layout::

    food-101/
      images/<class>/<image>.jpg
      meta/train.txt
      meta/test.txt

This script keeps the official test split and creates a stratified validation
split from the training set so the benchmark has train/val/test subsets.
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

from getitune.benchmark.dataset_helpers import download_kaggle_dataset, parse_args, resolve_raw_source

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

_KAGGLE_DATASET_ID = "kmader/food41"
_TRAIN_SPLIT_RATIO = 0.9


def _find_food_root(root: Path) -> Path:
    """Find the Food-101 root directory under *root*."""
    for train_txt in root.rglob("train.txt"):
        for candidate in (train_txt.parent, train_txt.parent.parent, train_txt.parent.parent.parent):
            if (candidate / "images").is_dir() and (candidate / "meta" / "train.txt").is_file():
                return candidate
            if (candidate / "images").is_dir() and (candidate / "meta" / "meta" / "train.txt").is_file():
                return candidate
    msg = f"Could not find a Food-101 dataset root under {root}"
    raise RuntimeError(msg)


def _read_split_lines(path: Path) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            label_name, image_name = line.strip().split("/", 1)
            entries.append((image_name, label_name))
    return entries


def _resolve_split_file(data_root: Path, split_name: str) -> Path:
    """Find a Food-101 split file in either the flat or nested Kaggle layout."""
    candidates = [
        data_root / "meta" / f"{split_name}.txt",
        data_root / "meta" / "meta" / f"{split_name}.txt",
        data_root / f"{split_name}.txt",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    msg = f"Could not find {split_name}.txt under {data_root}"
    raise RuntimeError(msg)


def _build_dataset(data_root: Path) -> Dataset:
    """Parse Food-101 split files and build a Datumaro dataset."""
    train_entries = _read_split_lines(_resolve_split_file(data_root, "train"))
    test_entries = _read_split_lines(_resolve_split_file(data_root, "test"))
    label_names = sorted({label for _, label in (*train_entries, *test_entries)})
    label_to_idx = {label: idx for idx, label in enumerate(label_names)}
    dataset: Dataset = Dataset(CocoSample, categories={"labels": CocoCategories(labels=label_names)})

    per_label: dict[str, list[tuple[str, str]]] = {}
    for image_name, label_name in train_entries:
        per_label.setdefault(label_name, []).append((image_name, label_name))

    train_examples: list[tuple[str, str]] = []
    val_examples: list[tuple[str, str]] = []
    for label_name in sorted(per_label):
        entries = per_label[label_name]
        val_count = max(1, round(len(entries) * (1 - _TRAIN_SPLIT_RATIO)))
        split_index = max(1, len(entries) - val_count)
        train_examples.extend(entries[:split_index])
        val_examples.extend(entries[split_index:])

    split_map: list[tuple[Subset, Iterable[tuple[str, str]]]] = [
        (Subset.TRAINING, train_examples),
        (Subset.VALIDATION, val_examples),
        (Subset.TESTING, test_entries),
    ]

    image_id = 0
    image_root = data_root / "images"
    for subset, entries in split_map:
        for image_name, label_name in entries:
            img_path = image_root / label_name / f"{image_name}.jpg"
            if not img_path.is_file():
                continue
            with Image.open(img_path) as im:
                width, height = im.size
            dataset.append(
                CocoSample(
                    image=LazyImage(img_path),
                    image_info=ImageInfo(width=width, height=height),
                    image_id=image_id,
                    subset=subset,
                    bboxes=None,
                    labels=np.asarray([label_to_idx[label_name]], dtype=np.int64),
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
    """Prepare the food41 benchmark dataset."""
    args = parse_args(description="Prepare the food41 benchmark dataset.")
    raw_root = resolve_raw_source(args, lambda: download_kaggle_dataset(_KAGGLE_DATASET_ID))
    data_root = _find_food_root(raw_root)
    dataset = _build_dataset(data_root)

    if args.dest.exists():
        shutil.rmtree(args.dest)
    args.dest.parent.mkdir(parents=True, exist_ok=True)
    export_dataset(dataset, args.dest)

    print(f"Dataset '{args.name}' ready at {args.dest}")


if __name__ == "__main__":
    main()
