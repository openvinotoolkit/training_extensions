# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""End-to-end test for ``scripts/benchmark_datasets/prepare_wgisd.py``.

This test performs a real network download of the WGISD dataset and is therefore
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
from datumaro.experimental.export_import import import_dataset
from datumaro.experimental.fields import Subset

# Path to the script under test.
_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "benchmark_datasets" / "prepare_wgisd.py"

# Total number of images in the upstream WGISD repo at the pinned commit.
_EXPECTED_TOTAL = 137


@pytest.mark.network
def test_prepare_wgisd_end_to_end(tmp_path: Path) -> None:
    """Run the prepare script end-to-end and validate the exported Datumaro dataset."""
    assert _SCRIPT.is_file(), f"Script not found: {_SCRIPT}"

    result = subprocess.run(
        [sys.executable, str(_SCRIPT), "--output-dir", str(tmp_path), "--name", "wgisd"],
        check=False,
        capture_output=True,
        text=True,
        timeout=600,  # 10 min: plenty of slack for the GitHub download.
    )
    assert result.returncode == 0, (
        f"Script failed (exit {result.returncode})\n--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )

    dataset_dir = tmp_path / "wgisd"
    assert dataset_dir.is_dir(), f"Expected output directory not found: {dataset_dir}"

    # The downloaded archive must be cleaned up.
    assert not (tmp_path / ".archives" / "wgisd.zip").exists()

    # ----- validate the exported dataset -----
    dataset = import_dataset(dataset_dir)
    assert len(dataset) == _EXPECTED_TOTAL

    counts = {Subset.TRAINING: 0, Subset.VALIDATION: 0, Subset.TESTING: 0}
    for sample in dataset:
        counts[sample.subset] += 1

    # Deterministic split: 88 / 22 / 27 (see prepare_wgisd._VAL_FRACTION).
    assert counts == {
        Subset.TRAINING: 88,
        Subset.VALIDATION: 22,
        Subset.TESTING: 27,
    }, f"Unexpected subset distribution: {counts}"
