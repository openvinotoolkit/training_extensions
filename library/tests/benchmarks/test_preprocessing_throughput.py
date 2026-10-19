# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Phase 0 measurement entry points.

Marked ``benchmark`` so they never run in the default unit-test sweep.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.benchmarks.harness import control_experiment, run_once, write_report
from tests.benchmarks.synthetic import DIAGONAL_SWEEP, make_synthetic_coco

from getitune.types import TaskType

pytestmark = pytest.mark.benchmark


@pytest.fixture(scope="session")
def report_dir(pytestconfig: pytest.Config) -> Path:
    """Directory that collects the Phase 0 JSON artifacts."""
    root = pytestconfig.getoption("--output-root") or ".benchmarks"
    path = Path(root) / "phase0"
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.mark.parametrize("cell", DIAGONAL_SWEEP, ids=lambda c: c.slug)
def test_synthetic_sweep(cell, tmp_path_factory, report_dir: Path) -> None:
    """Attribute preprocessing cost across image size x instances x vertices."""
    root = make_synthetic_coco(tmp_path_factory.mktemp(cell.slug), cell)
    result = run_once(
        root,
        TaskType.INSTANCE_SEGMENTATION,
        input_size=(cell.image_size, cell.image_size),
        batch_size=4,
        num_workers=0,  # serial ground truth: attribution sums cleanly
        max_batches=cell.num_images // 4,
    )
    write_report(
        {"cell": cell.__dict__, "result": result.__dict__},
        report_dir / f"sweep_{cell.slug}.json",
    )
    # Coverage gate, not a performance gate: if the attribution table explains
    # less than 80% of wall time the remaining cost is usually allocator
    # pressure or IPC serialization of large mask tensors across the worker
    # boundary -- check tensor sizes crossing _default_collate_fn.
    assert result.coverage >= 0.80, (
        f"attribution covers only {result.coverage:.0%} of wall time; table={result.attribution}"
    )


@pytest.mark.network
def test_control_experiment(pytestconfig: pytest.Config, report_dir: Path) -> None:
    """The detection-vs-instance-seg ratio on real large images.

    Requires ``satellite_building`` (9,665 aerial images, ``size_tier: large``,
    compatible with both ``detection`` and ``instance_segmentation``):

        uv run python -m getitune.benchmark provision --dataset satellite_building
    """
    data_root = Path(pytestconfig.getoption("--data-root")) / "satellite_building"
    if not data_root.exists():
        pytest.skip(f"{data_root} not prepared")

    payload = control_experiment(data_root, input_size=(1024, 1024), batch_size=8, num_workers=8)
    write_report(payload, report_dir / "control_experiment.json")

    assert payload["ratio"] > 0.0
