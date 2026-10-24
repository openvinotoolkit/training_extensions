# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for the Roboflow-100 HuggingFace detection prepare scripts.

These tests perform a real network download of the upstream parquet splits and
are therefore **marked with** ``@pytest.mark.network``. They are skipped by
default and only run when pytest is invoked with ``--run-network``.

In CI, the tests are gated behind a path filter so they only run on PRs that
touch ``library/scripts/benchmark_datasets/**`` (see
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

_SCRIPTS_DIR = Path(__file__).resolve().parents[3] / "scripts" / "benchmark_datasets"


# Each case: (dataset name, script file, expected subset counts, expected labels).
_CASES = {
    "axial_mri": (
        "prepare_axial_mri.py",
        {Subset.TRAINING: 253, Subset.VALIDATION: 39, Subset.TESTING: 79},
        ("axial-MRI", "negative", "positive"),
    ),
    "cable_damage": (
        "prepare_cable_damage.py",
        {Subset.TRAINING: 919, Subset.VALIDATION: 134, Subset.TESTING: 265},
        ("cable-damage", "break", "thunderbolt"),
    ),
    "vehicles": (
        "prepare_vehicles.py",
        {Subset.TRAINING: 2634, Subset.VALIDATION: 458, Subset.TESTING: 966},
        (
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
        ),
    ),
    "pothole": (
        "prepare_pothole.py",
        {Subset.TRAINING: 80, Subset.VALIDATION: 5, Subset.TESTING: 5},
        ("pothole",),
    ),
    "pcb_defect": (
        "prepare_pcb_defect.py",
        {Subset.TRAINING: 128, Subset.VALIDATION: 25, Subset.TESTING: 36},
        ("dry_joint", "incorrect_installation", "pcb_damage", "short_circuit"),
    ),
    "satellite_building": (
        "prepare_satellite_building.py",
        {Subset.TRAINING: 6764, Subset.VALIDATION: 1934, Subset.TESTING: 967},
        ("building",),
    ),
}


@pytest.mark.network
@pytest.mark.parametrize("name", list(_CASES))
def test_prepare_roboflow_hf_script_end_to_end(name: str, tmp_path: Path) -> None:
    """Run a prepare script end-to-end and validate the exported Datumaro dataset."""
    script_file, expected_counts, expected_labels = _CASES[name]
    script = _SCRIPTS_DIR / script_file
    assert script.is_file(), f"Script not found: {script}"

    result = subprocess.run(
        [sys.executable, str(script), "--output-dir", str(tmp_path), "--name", name],
        check=False,
        capture_output=True,
        text=True,
        timeout=1800,  # 30 min: the largest parquet (vehicles) is several hundred MB.
    )
    assert result.returncode == 0, (
        f"Script failed (exit {result.returncode})\n--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )

    dataset_dir = tmp_path / name
    assert dataset_dir.is_dir(), f"Expected output directory not found: {dataset_dir}"

    # The downloaded parquet splits must be cleaned up.
    archives = tmp_path / ".archives"
    for split in ("train", "validation", "test"):
        assert not (archives / f"{name}_{split}.parquet").exists()

    # ----- validate the exported dataset -----
    dataset = import_dataset(dataset_dir)
    expected_total = sum(expected_counts.values())
    assert len(dataset) == expected_total

    label_categories = dataset.label_categories
    assert isinstance(label_categories, LabelCategories)
    assert tuple(label_categories.labels) == expected_labels

    counts = {Subset.TRAINING: 0, Subset.VALIDATION: 0, Subset.TESTING: 0}
    for sample in dataset:
        counts[sample.subset] += 1
    assert counts == expected_counts, f"Unexpected subset distribution: {counts}"

    # The exported dataset must convert to the getitune detection schema, which
    # only accepts COCO-style ``xywh`` boxes.
    detection_dataset = dataset.convert_to_schema(DetectionSample)
    samples_with_boxes = 0
    total_boxes = 0
    for sample in detection_dataset:
        n_boxes = 0 if sample.bboxes is None else len(sample.bboxes)
        if n_boxes:
            samples_with_boxes += 1
            total_boxes += n_boxes

    assert samples_with_boxes > expected_total // 2, (
        f"Expected most images to carry annotations, got {samples_with_boxes}/{expected_total}"
    )
    assert total_boxes > samples_with_boxes, "Expected multiple bounding boxes across the dataset"
