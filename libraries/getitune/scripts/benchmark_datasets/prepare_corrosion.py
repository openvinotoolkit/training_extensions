#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Download and prepare the 'corrosion' benchmark dataset (tiny / semantic segmentation).

Downloads a corrosion condition-state segmentation dataset from its public
HuggingFace mirror and exports it in the experimental Datumaro format. Each
image ships a real per-pixel mask labelling corrosion severity across 4
classes (``nothing``, ``fair``, ``poor``, ``severe``), making it a small,
real-world multi-class semantic-segmentation benchmark.

The upstream mask uses a small VOC-style RGB palette (one flat colour per
class: ``(0, 0, 0)``, ``(128, 0, 0)``, ``(0, 128, 0)``, ``(128, 128, 0)``),
which is mapped to class indices here.

The dataset ships official ``train`` / ``validation`` splits only (44 images
total, no ``test`` split); all images are merged and deterministically
re-split 4/6-1/6-1/6 (train/validation/test) here so the exported dataset has
all three subsets.

Mirror: https://huggingface.co/datasets/rkumari/corrosion_segment
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
_REPO = "rkumari/corrosion_segment"
_REVISION = "1cf5c86835f0670e0532a31e1dc2f2675df11d04"

# Matches the upstream ``id2label.json`` exactly (index 0 is the "no corrosion" class).
_LABEL_NAMES = ("nothing", "fair", "poor", "severe")

_SPLIT_FILES = {
    "train": "data/train-00000-of-00001-f1fc47ca63c5f418.parquet",
    "validation": "data/validation-00000-of-00001-913b0c6bbddc79df.parquet",
}

# VOC-style flat-colour palette -> class index, verified against the raw dataset bytes.
_COLOR_TO_CLASS: dict[tuple[int, int, int], int] = {
    (0, 0, 0): 0,  # nothing
    (128, 0, 0): 1,  # fair
    (0, 128, 0): 2,  # poor
    (128, 128, 0): 3,  # severe
}


def _decode_mask(raw_mask: PILImage.Image) -> np.ndarray:
    """Map the dataset's flat VOC-style RGB palette to class indices."""
    arr = np.array(raw_mask.convert("RGB"))
    out = np.zeros(arr.shape[:2], dtype=np.uint8)
    for color, class_idx in _COLOR_TO_CLASS.items():
        out[np.all(arr == np.asarray(color, dtype=np.uint8), axis=-1)] = class_idx
    return out


def _assign_subset(_split: str, index: int) -> Subset:
    """Merge both upstream splits and re-split ~4/6-1/6-1/6, interleaved for balance.

    Upstream ships only ``train``/``validation`` (no ``test``), and with only 44
    images total a contiguous slice risks a fold with zero instances of the rarer
    severity classes, so rows are interleaved by their global position instead.
    """
    remainder = index % 6
    if remainder == 0:
        return Subset.TESTING
    if remainder == 1:
        return Subset.VALIDATION
    return Subset.TRAINING


def main() -> None:
    """Download the corrosion segmentation dataset, convert to Datumaro format, and save it."""
    args = parse_args(description="Prepare the corrosion benchmark dataset.")
    prepare_hf_mask_dataset(
        args,
        repo=_REPO,
        revision=_REVISION,
        split_files=_SPLIT_FILES,
        label_names=_LABEL_NAMES,
        image_column="image",
        mask_column="label",
        mask_decoder=_decode_mask,
        assign_subset=_assign_subset,
    )


if __name__ == "__main__":
    main()
