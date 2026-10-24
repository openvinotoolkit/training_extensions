#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Download and prepare the 'crack_semantic' benchmark dataset (large / semantic segmentation).

Downloads the "Crack Segmentation Dataset" (a widely-used public compilation of
several pavement/wall crack datasets, e.g. CFD, CRACK500, DeepCrack, GAPS384)
from its public HuggingFace mirror and exports it in the experimental Datumaro
format. Each image ships a real per-pixel binary mask (``background`` /
``crack``), making it a large-scale semantic-segmentation benchmark.

The upstream mask is a near-binary grayscale image with antialiasing noise at
crack boundaries (introduced by an upstream resize step) rather than clean
``{0, 255}`` values, so it is thresholded here into clean class indices.

The dataset ships official ``train`` / ``test`` splits only; a validation
subset is carved deterministically out of ``train`` (every 7th image, in
parquet row order) so the exported dataset has all three subsets.

Mirror: https://huggingface.co/datasets/varcoder/crack-segmentation-dataset
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from datumaro.experimental.fields import Subset
from hf_mask_helper import prepare_hf_mask_dataset

from getitune.benchmark.dataset_helpers import parse_args

if TYPE_CHECKING:
    from PIL import Image as PILImage

# Pinned commit of the mirror's ``main`` branch for reproducibility.
_REPO = "varcoder/crack-segmentation-dataset"
_REVISION = "70fafe2ec3c70f48ba5f76c6f10ae2f539bcc44b"

_LABEL_NAMES = ("background", "crack")

_SPLIT_FILES = {
    "train": (
        "data/train-00000-of-00003-06e580b36935d137.parquet",
        "data/train-00001-of-00003-b7d4634b852df18c.parquet",
        "data/train-00002-of-00003-d7c3d9a5fc0295d1.parquet",
    ),
    "test": ("data/test-00000-of-00001-3da1883d5c5df420.parquet",),
}

# Every Nth training image (in parquet row order) is carved out as validation.
_VAL_STEP = 7

# Pixel values at/above this threshold are treated as "crack" foreground; the raw
# mask has antialiasing noise near boundaries rather than clean {0, 255} values.
_THRESHOLD = 128


def _decode_mask(raw_mask: PILImage.Image) -> np.ndarray:
    """Binarize the near-binary grayscale crack mask."""
    arr = np.array(raw_mask.convert("L"))
    return (arr >= _THRESHOLD).astype(np.uint8)


def _assign_subset(split: str, index: int) -> Subset:
    """Keep the official test split; carve validation out of train deterministically."""
    if split == "test":
        return Subset.TESTING
    return Subset.VALIDATION if index % _VAL_STEP == 0 else Subset.TRAINING


def main() -> None:
    """Download the Crack Segmentation Dataset, convert to Datumaro format, and save it."""
    args = parse_args(description="Prepare the crack_semantic benchmark dataset.")
    prepare_hf_mask_dataset(
        args,
        repo=_REPO,
        revision=_REVISION,
        split_files=_SPLIT_FILES,
        label_names=_LABEL_NAMES,
        image_column="pixel_values",
        mask_column="label",
        mask_decoder=_decode_mask,
        assign_subset=_assign_subset,
    )


if __name__ == "__main__":
    main()
