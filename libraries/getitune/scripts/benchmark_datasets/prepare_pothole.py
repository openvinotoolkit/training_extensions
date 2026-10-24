#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Download and prepare the 'pothole' benchmark dataset (tiny / instance segmentation).

Downloads the "Pothole Segmentation" Roboflow Universe project via its public
HuggingFace mirror (``keremberke/pothole-segmentation``, produced by the
community ``roboflow2huggingface`` tool) and exports it in the experimental
Datumaro format. The dataset contains road-surface images with polygon
instance-segmentation annotations for a single ``pothole`` class, making it a
small, real-world instance-segmentation benchmark.

Source: https://universe.roboflow.com/roboflow-100/pothole-segmentation
Mirror: https://huggingface.co/datasets/keremberke/pothole-segmentation
"""

from __future__ import annotations

from roboflow_hf_helper import prepare_roboflow_hf_dataset

from getitune.benchmark.dataset_helpers import parse_args

# Pinned commit of the mirror's auto-converted-parquet ref
# (``refs/convert/parquet``) for reproducibility.
_REPO = "keremberke/pothole-segmentation"
_REVISION = "0dd5d9d3278e65950f78657399f6aff2aa680a46"

# Class names in upstream ``ClassLabel`` index order. Unlike the official
# Roboflow-100 ``Francesco/*`` mirrors, ``keremberke/*`` conversions do not carry a
# super-category placeholder at index 0.
_LABEL_NAMES = ("pothole",)

_SPLIT_FILES = {
    "train": "full/train/0000.parquet",
    "validation": "full/validation/0000.parquet",
    "test": "full/test/0000.parquet",
}


def main() -> None:
    """Download Pothole Segmentation, convert it to the experimental Datumaro format, and save it."""
    args = parse_args(description="Prepare the pothole benchmark dataset.")
    prepare_roboflow_hf_dataset(
        args,
        repo=_REPO,
        revision=_REVISION,
        label_names=_LABEL_NAMES,
        split_files=_SPLIT_FILES,
        include_segmentation=True,
    )


if __name__ == "__main__":
    main()
