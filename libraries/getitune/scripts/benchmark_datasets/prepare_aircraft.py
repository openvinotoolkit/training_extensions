#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Download and prepare the FGVC Aircraft benchmark dataset.

Uses the Kaggle mirror of the FGVC-Aircraft benchmark and exports it in the
experimental Datumaro format for multi-class classification benchmarks.

Source: https://www.kaggle.com/datasets/seryouxblaster764/fgvc-aircraft

The official FGVC-Aircraft release organizes the data as a flat ``data/``
directory with text split files such as ``images_variant_train.txt`` and image
paths under ``data/images``. This script supports both automatic Kaggle
download and ``--raw-dir`` pointing at a pre-extracted dataset root.
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
    from pathlib import Path

_KAGGLE_DATASET_ID = "seryouxblaster764/fgvc-aircraft"

_LABEL_LEVELS = {
    "variant": "images_variant_{split}.txt",
    "family": "images_family_{split}.txt",
    "manufacturer": "images_manufacturer_{split}.txt",
}

_SPLITS = {
    "train": Subset.TRAINING,
    "val": Subset.VALIDATION,
    "test": Subset.TESTING,
}


def _find_data_root(root: Path) -> Path:
    """Return the FGVC-Aircraft ``data`` directory under *root*."""
    for candidate in (root, *root.rglob("data")):
        if (
            candidate.is_dir()
            and (candidate / "images").is_dir()
            and (candidate / "images_variant_train.txt").is_file()
        ):
            return candidate
    msg = f"Could not find an FGVC-Aircraft data root under {root}"
    raise RuntimeError(msg)


def _load_label_names(data_root: Path, annotation_level: str) -> list[str]:
    label_file = data_root / f"{_LABEL_LEVELS[annotation_level]}".format(split="train")
    labels: list[str] = []
    with label_file.open(encoding="utf-8") as f:
        for line in f:
            _, label = line.strip().split(" ", 1)
            labels.append(label)
    return sorted(set(labels))


def _build_dataset(data_root: Path, annotation_level: str) -> Dataset:
    """Parse FGVC-Aircraft split files and build a Datumaro dataset."""
    label_names = _load_label_names(data_root, annotation_level)
    label_to_idx = {label: idx for idx, label in enumerate(label_names)}
    dataset: Dataset = Dataset(CocoSample, categories={"labels": CocoCategories(labels=label_names)})

    image_root = data_root / "images"
    image_id = 0
    for split_name, subset in _SPLITS.items():
        split_file = data_root / _LABEL_LEVELS[annotation_level].format(split=split_name)
        with split_file.open(encoding="utf-8") as f:
            for line in f:
                image_name, label_name = line.strip().split(" ", 1)
                img_path = image_root / f"{image_name}.jpg"
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
    """Prepare the aircraft benchmark dataset."""
    args = parse_args(description="Prepare the aircraft benchmark dataset.")
    raw_root = resolve_raw_source(args, lambda: download_kaggle_dataset(_KAGGLE_DATASET_ID))
    data_root = _find_data_root(raw_root)
    dataset = _build_dataset(data_root, annotation_level="variant")

    if args.dest.exists():
        shutil.rmtree(args.dest)
    args.dest.parent.mkdir(parents=True, exist_ok=True)
    export_dataset(dataset, args.dest)

    print(f"Dataset '{args.name}' ready at {args.dest}")


if __name__ == "__main__":
    main()
