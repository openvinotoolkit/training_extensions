# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""End-to-end test for ``scripts/benchmark_datasets/prepare_flowers102.py``.

This test performs a real network download and is therefore marked with
``@pytest.mark.network``. It is skipped by default and only runs when pytest
is invoked with ``--run-network``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from datumaro.experimental.export_import import import_dataset
from datumaro.experimental.fields import Subset

_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "benchmark_datasets" / "prepare_flowers102.py"

# Oxford Flowers 102 has 8189 images total.
_EXPECTED_TOTAL = 8189


@pytest.mark.network
def test_prepare_flowers102_end_to_end(tmp_path: Path) -> None:
    """Run the prepare script end-to-end and validate the exported Datumaro dataset."""
    assert _SCRIPT.is_file(), f"Script not found: {_SCRIPT}"

    result = subprocess.run(
        [sys.executable, str(_SCRIPT), "--output-dir", str(tmp_path), "--name", "flowers102"],
        check=False,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, (
        f"Script failed (exit {result.returncode})\n--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )

    dataset_dir = tmp_path / "flowers102"
    assert dataset_dir.is_dir(), f"Expected output directory not found: {dataset_dir}"

    # Validate the exported dataset
    dataset = import_dataset(dataset_dir)
    assert len(dataset) == _EXPECTED_TOTAL

    counts = {Subset.TRAINING: 0, Subset.VALIDATION: 0, Subset.TESTING: 0}
    for sample in dataset:
        counts[sample.subset] += 1

    # Official splits: 1020 train, 1020 val, 6149 test
    assert counts[Subset.TRAINING] == 1020
    assert counts[Subset.VALIDATION] == 1020
    assert counts[Subset.TESTING] == 6149
