# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""End-to-end test for ``scripts/benchmark_datasets/prepare_aid_multilabel.py``.

This test performs a real network download of the AID Multi-Label dataset and is
therefore **marked with** ``@pytest.mark.network``. It is skipped by default and
only runs when pytest is invoked with ``--run-network``.

In CI, the test is gated behind a path filter so it only runs on PRs that touch
``library/scripts/benchmark_datasets/**`` (see
``.github/workflows/lib-lint-and-test.yaml``).
"""

from __future__ import annotations

import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest
from datumaro.experimental.categories import LabelCategories
from datumaro.experimental.data_formats.coco.sample import CocoSample
from datumaro.experimental.export_import import import_dataset
from datumaro.experimental.fields import Subset

# Path to the script under test.
_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "benchmark_datasets" / "prepare_aid_multilabel.py"

# Total number of images in the upstream AID Multi-Label parquet.
_EXPECTED_TOTAL = 3000

# Deterministic subset distribution produced by the script's index-based split.
_EXPECTED_COUNTS = {
    Subset.TRAINING: 2100,
    Subset.VALIDATION: 450,
    Subset.TESTING: 450,
}

# Number of multi-label classes defined by the dataset.
_EXPECTED_NUM_LABELS = 17


@pytest.mark.network
def test_prepare_aid_multilabel_end_to_end(tmp_path: Path) -> None:
    """Run the prepare script end-to-end and validate the exported Datumaro dataset."""
    assert _SCRIPT.is_file(), f"Script not found: {_SCRIPT}"

    result = subprocess.run(
        [sys.executable, str(_SCRIPT), "--output-dir", str(tmp_path), "--name", "aid_multilabel"],
        check=False,
        capture_output=True,
        text=True,
        timeout=900,  # 15 min: the parquet is ~270 MB.
    )
    assert result.returncode == 0, (
        f"Script failed (exit {result.returncode})\n--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )

    dataset_dir = tmp_path / "aid_multilabel"
    assert dataset_dir.is_dir(), f"Expected output directory not found: {dataset_dir}"

    # The downloaded parquet must be cleaned up.
    assert not (tmp_path / ".archives" / "aid_multilabel.parquet").exists()
    # Images must have been copied into the exported dataset.
    assert len(list((dataset_dir / "images").glob("*.jpg"))) == _EXPECTED_TOTAL

    # ----- validate the exported dataset -----
    dataset = import_dataset(dataset_dir).convert_to_schema(CocoSample)
    assert len(dataset) == _EXPECTED_TOTAL
    label_categories = dataset.label_categories
    assert isinstance(label_categories, LabelCategories)
    assert len(label_categories.labels) == _EXPECTED_NUM_LABELS

    counts: Counter[Subset] = Counter()
    multi_label_images = 0
    used_labels: set[int] = set()
    for sample in dataset:
        counts[sample.subset] += 1
        labels = sample.labels.tolist() if sample.labels is not None else []
        used_labels.update(int(label) for label in labels)
        if len(labels) > 1:
            multi_label_images += 1

    assert dict(counts) == _EXPECTED_COUNTS, f"Unexpected subset distribution: {dict(counts)}"

    # All 17 classes must appear, and the dataset must be genuinely multi-label
    # (the overwhelming majority of images carry more than one label).
    assert len(used_labels) == _EXPECTED_NUM_LABELS
    assert multi_label_images > _EXPECTED_TOTAL // 2, (
        f"Expected a genuine multi-label dataset, only {multi_label_images}/{_EXPECTED_TOTAL} images had >1 label"
    )
