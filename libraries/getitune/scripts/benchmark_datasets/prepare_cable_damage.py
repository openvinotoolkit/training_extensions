#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Download and prepare the 'cable_damage' benchmark dataset (medium / inspection).

Downloads the Roboflow-100 "Cable Damage" dataset from its public HuggingFace
mirror and exports it in the experimental Datumaro format for detection
benchmarks. The dataset contains close-up images of power/transmission cables
annotated with damage bounding boxes (``break`` / ``thunderbolt``), a
representative industrial damage-inspection detection benchmark.

Source: https://universe.roboflow.com/roboflow-100/cable-damage
Mirror: https://huggingface.co/datasets/Francesco/cable-damage
"""

from __future__ import annotations

from roboflow_hf_helper import prepare_roboflow_hf_dataset

from getitune.benchmark.dataset_helpers import parse_args

# Pinned revision of the HuggingFace mirror for reproducibility.
_REPO = "Francesco/cable-damage"
_REVISION = "b598e4ecc1e6492450ca86af9f85867d395ecd95"

# Class names in upstream ``ClassLabel`` index order. Index 0 is the Roboflow
# super-category placeholder that never appears in annotations.
_LABEL_NAMES = ("cable-damage", "break", "thunderbolt")

_SPLIT_FILES = {
    "train": "data/train-00000-of-00001-8eedf0718cda8fe2.parquet",
    "validation": "data/validation-00000-of-00001-34f26f83cd59a188.parquet",
    "test": "data/test-00000-of-00001-b85b52944122e167.parquet",
}


def main() -> None:
    """Download Cable Damage, convert it to the experimental Datumaro format, and save it."""
    args = parse_args(description="Prepare the cable_damage benchmark dataset.")
    prepare_roboflow_hf_dataset(
        args,
        repo=_REPO,
        revision=_REVISION,
        label_names=_LABEL_NAMES,
        split_files=_SPLIT_FILES,
    )


if __name__ == "__main__":
    main()
