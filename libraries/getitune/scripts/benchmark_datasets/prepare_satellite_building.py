#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Download and prepare the 'satellite_building' benchmark dataset (medium / semantic segmentation).

Downloads the "Satellite Building Segmentation" Roboflow Universe project via
its public HuggingFace mirror (``keremberke/satellite-building-segmentation``,
produced by the community ``roboflow2huggingface`` tool) and exports it in the
experimental Datumaro format. The dataset contains aerial/satellite imagery
with polygon instance-segmentation annotations for a single ``building``
class (building-footprint extraction), making it a mid-scale real-world
semantic-segmentation benchmark.

The dataset is exported with its native polygons intact (as a ``CocoSample``):
at training time, Datumaro's schema conversion automatically rasterizes those
polygons into precise per-pixel masks for the semantic-segmentation benchmark
(verified to rasterize the true polygon shape, not a bounding-box
approximation) — no manual mask pre-computation is needed here.

Source: https://universe.roboflow.com/roboflow-100/satellite-building-segmentation
Mirror: https://huggingface.co/datasets/keremberke/satellite-building-segmentation
"""

from __future__ import annotations

from roboflow_hf_helper import prepare_roboflow_hf_dataset

from getitune.benchmark.dataset_helpers import parse_args

# Pinned commit of the mirror's auto-converted-parquet ref
# (``refs/convert/parquet``) for reproducibility.
_REPO = "keremberke/satellite-building-segmentation"
_REVISION = "e024f12bc5dc293f901e726ef330bee3c9dafaf3"

# Class names in upstream ``ClassLabel`` index order. Unlike the official
# Roboflow-100 ``Francesco/*`` mirrors, ``keremberke/*`` conversions do not carry a
# super-category placeholder at index 0.
_LABEL_NAMES = ("building",)

_SPLIT_FILES = {
    "train": "full/train/0000.parquet",
    "validation": "full/validation/0000.parquet",
    "test": "full/test/0000.parquet",
}


def main() -> None:
    """Download Satellite Building Segmentation, convert to the experimental Datumaro format, and save it."""
    args = parse_args(description="Prepare the satellite_building benchmark dataset.")
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
