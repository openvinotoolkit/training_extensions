#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Download and prepare the 'axial_mri' benchmark dataset (tiny / medical).

Downloads the Roboflow-100 "Axial MRI" dataset from its public HuggingFace
mirror and exports it in the experimental Datumaro format for detection
benchmarks. The dataset contains brain axial-MRI slices annotated with tumour
bounding boxes (``negative`` / ``positive``), making it a small, privacy-safe
medical detection benchmark.

Source: https://universe.roboflow.com/roboflow-100/axial-mri
Mirror: https://huggingface.co/datasets/Francesco/axial-mri
"""

from __future__ import annotations

from roboflow_hf_helper import prepare_roboflow_hf_dataset

from getitune.benchmark.dataset_helpers import parse_args

# Pinned revision of the HuggingFace mirror for reproducibility.
_REPO = "Francesco/axial-mri"
_REVISION = "0b502f037865f4cf9ec1254a8c34fc3f19caaeef"

# Class names in upstream ``ClassLabel`` index order. Index 0 is the Roboflow
# super-category placeholder that never appears in annotations.
_LABEL_NAMES = ("axial-MRI", "negative", "positive")

_SPLIT_FILES = {
    "train": "data/train-00000-of-00001-62cf6bf015fef032.parquet",
    "validation": "data/validation-00000-of-00001-bcd8291312ff472b.parquet",
    "test": "data/test-00000-of-00001-7780878af8cf3e7b.parquet",
}


def main() -> None:
    """Download Axial MRI, convert it to the experimental Datumaro format, and save it."""
    args = parse_args(description="Prepare the axial_mri benchmark dataset.")
    prepare_roboflow_hf_dataset(
        args,
        repo=_REPO,
        revision=_REVISION,
        label_names=_LABEL_NAMES,
        split_files=_SPLIT_FILES,
    )


if __name__ == "__main__":
    main()
