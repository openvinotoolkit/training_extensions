# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""End-to-end test for ``scripts/benchmark_datasets/prepare_bccd.py``.

This test performs a real network download of the BCCD dataset and is therefore
**marked with** ``@pytest.mark.network``. It is skipped by default and only runs
when pytest is invoked with ``--run-network``.

In CI, the test is gated behind a path filter so it only runs on PRs that touch
``library/scripts/benchmark_datasets/**`` (see
``.github/workflows/lib-lint-and-test.yaml``).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from datumaro.experimental.categories import LabelCategories
from datumaro.experimental.export_import import import_dataset
from datumaro.experimental.fields import Subset

from getitune.data.entity.sample import DetectionSample

# Path to the script under test.
_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "benchmark_datasets" / "prepare_bccd.py"

# Total number of images in the upstream BCCD repo at the pinned commit.
_EXPECTED_TOTAL = 364

# Deterministic subset distribution from BCCD's official ImageSets/Main split
# files (train.txt / val.txt / test.txt).
_EXPECTED_COUNTS = {
    Subset.TRAINING: 205,
    Subset.VALIDATION: 87,
    Subset.TESTING: 72,
}

# Fixed label set defined by the dataset.
_EXPECTED_LABELS = ("Platelets", "RBC", "WBC")


@pytest.mark.network
def test_prepare_bccd_end_to_end(tmp_path: Path) -> None:
    """Run the prepare script end-to-end and validate the exported Datumaro dataset."""
    assert _SCRIPT.is_file(), f"Script not found: {_SCRIPT}"

    result = subprocess.run(
        [sys.executable, str(_SCRIPT), "--output-dir", str(tmp_path), "--name", "bccd"],
        check=False,
        capture_output=True,
        text=True,
        timeout=600,  # 10 min: plenty of slack for the GitHub download.
    )
    assert result.returncode == 0, (
        f"Script failed (exit {result.returncode})\n--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )

    dataset_dir = tmp_path / "bccd"
    assert dataset_dir.is_dir(), f"Expected output directory not found: {dataset_dir}"

    # The downloaded archive must be cleaned up.
    assert not (tmp_path / ".archives" / "bccd.zip").exists()

    # ----- validate the exported dataset -----
    dataset = import_dataset(dataset_dir)
    assert len(dataset) == _EXPECTED_TOTAL
    label_categories = dataset.label_categories
    assert isinstance(label_categories, LabelCategories)
    assert tuple(label_categories.labels) == _EXPECTED_LABELS

    counts = {Subset.TRAINING: 0, Subset.VALIDATION: 0, Subset.TESTING: 0}
    for sample in dataset:
        counts[sample.subset] += 1
    assert counts == _EXPECTED_COUNTS, f"Unexpected subset distribution: {counts}"

    # The exported dataset must convert to the getitune detection schema, which
    # only accepts COCO-style boxes (the dataset is therefore exported as
    # ``CocoSample``/``xywh``, not VOC ``xyxy``). This mirrors what
    # ``DetectionDataset`` does during training and guards the regression where
    # a VOC ``xyxy`` export failed with "Input format 'xyxy' is not supported".
    detection_dataset = dataset.convert_to_schema(DetectionSample)
    samples_with_boxes = 0
    total_boxes = 0
    for sample in detection_dataset:
        n_boxes = 0 if sample.bboxes is None else len(sample.bboxes)
        if n_boxes:
            samples_with_boxes += 1
            total_boxes += n_boxes

    assert samples_with_boxes == _EXPECTED_TOTAL, (
        f"Expected every image to carry annotations, got {samples_with_boxes}/{_EXPECTED_TOTAL}"
    )
    assert total_boxes > _EXPECTED_TOTAL, "Expected multiple bounding boxes across the dataset"
