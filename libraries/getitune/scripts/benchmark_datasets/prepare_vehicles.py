#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Download and prepare the 'vehicles' benchmark dataset (large / automotive).

Downloads the Roboflow-100 "Vehicles" dataset from its public HuggingFace
mirror and exports it in the experimental Datumaro format for detection
benchmarks. The dataset contains road-scene images annotated with bounding
boxes across a dozen vehicle types (cars, buses, trucks of various sizes),
making it a large multi-class on-road detection benchmark.

Source: https://universe.roboflow.com/roboflow-100/vehicles-q0x2v
Mirror: https://huggingface.co/datasets/Francesco/vehicles-q0x2v
"""

from __future__ import annotations

from roboflow_hf_helper import prepare_roboflow_hf_dataset

from getitune.benchmark.dataset_helpers import parse_args

# Pinned revision of the HuggingFace mirror for reproducibility.
_REPO = "Francesco/vehicles-q0x2v"
_REVISION = "c8685b5cdb5d505fe9e79286a8c532729bcac470"

# Class names in upstream ``ClassLabel`` index order. Index 0 is the Roboflow
# super-category placeholder that never appears in annotations.
_LABEL_NAMES = (
    "vehicles",
    "big bus",
    "big truck",
    "bus-l-",
    "bus-s-",
    "car",
    "mid truck",
    "small bus",
    "small truck",
    "truck-l-",
    "truck-m-",
    "truck-s-",
    "truck-xl-",
)

_SPLIT_FILES = {
    "train": "data/train-00000-of-00001-9656595db06ccb29.parquet",
    "validation": "data/validation-00000-of-00001-ca8cc590e7de3d43.parquet",
    "test": "data/test-00000-of-00001-033568a2b9f13ea2.parquet",
}


def main() -> None:
    """Download Vehicles, convert it to the experimental Datumaro format, and save it."""
    args = parse_args(description="Prepare the vehicles benchmark dataset.")
    prepare_roboflow_hf_dataset(
        args,
        repo=_REPO,
        revision=_REVISION,
        label_names=_LABEL_NAMES,
        split_files=_SPLIT_FILES,
    )


if __name__ == "__main__":
    main()
