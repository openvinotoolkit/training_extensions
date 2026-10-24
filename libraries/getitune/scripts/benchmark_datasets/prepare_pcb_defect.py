#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Download and prepare the 'pcb_defect' benchmark dataset (small / semantic segmentation).

Downloads the "PCB Defect Segmentation" Roboflow Universe project via its public
HuggingFace mirror (``keremberke/pcb-defect-segmentation``, produced by the
community ``roboflow2huggingface`` tool) and exports it in the experimental
Datumaro format. The dataset contains close-up PCB images with polygon
instance-segmentation annotations across 4 defect classes (dry joint,
incorrect installation, PCB damage, short circuit).

The dataset is exported with its native polygons intact (as a ``CocoSample``):
at training time, Datumaro's schema conversion automatically rasterizes those
polygons into precise per-pixel masks for the semantic-segmentation benchmark
(verified to rasterize the true polygon shape, not a bounding-box
approximation) — no manual mask pre-computation is needed here.

Source: https://universe.roboflow.com/roboflow-100/pcb-defect-segmentation
Mirror: https://huggingface.co/datasets/keremberke/pcb-defect-segmentation
"""

from __future__ import annotations

from roboflow_hf_helper import prepare_roboflow_hf_dataset

from getitune.benchmark.dataset_helpers import parse_args

# Pinned commit of the mirror's auto-converted-parquet ref
# (``refs/convert/parquet``) for reproducibility.
_REPO = "keremberke/pcb-defect-segmentation"
_REVISION = "7d0c8c6acf4613c539f063909b4223a7803f2d10"

# Class names in upstream ``ClassLabel`` index order. Unlike the official
# Roboflow-100 ``Francesco/*`` mirrors, ``keremberke/*`` conversions do not carry a
# super-category placeholder at index 0.
_LABEL_NAMES = ("dry_joint", "incorrect_installation", "pcb_damage", "short_circuit")

_SPLIT_FILES = {
    "train": "full/train/0000.parquet",
    "validation": "full/validation/0000.parquet",
    "test": "full/test/0000.parquet",
}


def main() -> None:
    """Download PCB Defect Segmentation, convert it to the experimental Datumaro format, and save it."""
    args = parse_args(description="Prepare the pcb_defect benchmark dataset.")
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
