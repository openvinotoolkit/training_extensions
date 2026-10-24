# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``scripts/benchmark_datasets/prepare_cardd.py``.

Two test tiers:

1. The ``--raw-dir`` path is exercised unconditionally (no network, no Kaggle
   credentials needed) against a small synthetic CarDD-style COCO export --
   proving that this credentialed dataset can be prepared entirely offline once
   its raw data has been fetched once. This is what runs on every PR.
2. The real Kaggle-download path is an end-to-end ``@pytest.mark.network``
   test, additionally skipped when Kaggle credentials aren't configured in the
   current environment.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from datumaro.experimental.export_import import import_dataset
from datumaro.experimental.fields import Subset
from PIL import Image
from scripts.benchmark_datasets.prepare_cardd import _build_dataset, _find_coco_root, _polygon_to_array

from getitune.benchmark.dataset_helpers import _has_kaggle_credentials

_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "benchmark_datasets" / "prepare_cardd.py"


def _make_synthetic_coco_dataset(root: Path) -> None:
    """Create a tiny CarDD-style COCO "split" export."""
    categories = [
        {"id": 1, "name": "dent", "supercategory": "car damages"},
        {"id": 2, "name": "scratch", "supercategory": "car damages"},
    ]

    def _write_split(split: str, images: list[dict], annotations: list[dict]) -> None:
        images_dir = root / split
        images_dir.mkdir(parents=True, exist_ok=True)
        for img in images:
            Image.new("RGB", (img["width"], img["height"]), color=(10, 20, 30)).save(images_dir / img["file_name"])
        payload = {
            "licenses": [],
            "info": {},
            "categories": categories,
            "images": images,
            "annotations": annotations,
        }
        (root / f"{split}.json").write_text(json.dumps(payload))

    _write_split(
        "train",
        images=[
            {"id": 1, "width": 100, "height": 100, "file_name": "000001.jpg"},
            {"id": 2, "width": 100, "height": 100, "file_name": "000002.jpg"},
        ],
        annotations=[
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "segmentation": [[25.0, 25.0, 75.0, 25.0, 75.0, 75.0, 25.0, 75.0]],
                "bbox": [25.0, 25.0, 50.0, 50.0],
                "area": 2500.0,
                "iscrowd": 0,
            },
            {
                "id": 2,
                "image_id": 2,
                "category_id": 2,
                "segmentation": [[10.0, 10.0, 40.0, 10.0, 40.0, 40.0, 10.0, 40.0]],
                "bbox": [10.0, 10.0, 30.0, 30.0],
                "area": 900.0,
                "iscrowd": 0,
            },
        ],
    )
    _write_split(
        "val",
        images=[{"id": 3, "width": 100, "height": 100, "file_name": "000003.jpg"}],
        annotations=[
            {
                "id": 3,
                "image_id": 3,
                "category_id": 1,
                "segmentation": [[5.0, 5.0, 20.0, 5.0, 20.0, 20.0, 5.0, 20.0]],
                "bbox": [5.0, 5.0, 15.0, 15.0],
                "area": 225.0,
                "iscrowd": 0,
            },
        ],
    )
    _write_split(
        "test",
        images=[{"id": 4, "width": 100, "height": 100, "file_name": "000004.jpg"}],
        annotations=[
            {
                "id": 4,
                "image_id": 4,
                "category_id": 2,
                "segmentation": [[1.0, 1.0, 9.0, 1.0, 9.0, 9.0, 1.0, 9.0]],
                "bbox": [1.0, 1.0, 8.0, 8.0],
                "area": 64.0,
                "iscrowd": 0,
            },
        ],
    )


class TestFindCocoRoot:
    def test_finds_coco_root(self, tmp_path: Path) -> None:
        nested = tmp_path / "outer" / "inner"
        _make_synthetic_coco_dataset(nested)
        assert _find_coco_root(tmp_path) == nested

    def test_missing_root_raises(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="No CarDD COCO-style dataset root found"):
            _find_coco_root(tmp_path)


class TestPolygonToArray:
    def test_converts_single_ring(self) -> None:
        polygon = _polygon_to_array([[0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0]])

        assert polygon.shape == (4, 2)
        assert polygon.dtype == np.float32
        assert polygon.tolist() == [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]


class TestBuildDataset:
    def test_builds_dataset_with_polygons_and_subsets(self, tmp_path: Path) -> None:
        _make_synthetic_coco_dataset(tmp_path)

        dataset = _build_dataset(tmp_path)

        assert len(dataset) == 4

        counts = {Subset.TRAINING: 0, Subset.VALIDATION: 0, Subset.TESTING: 0}
        for sample in dataset:
            counts[sample.subset] += 1
            assert sample.polygons is not None
            assert sample.polygons[0].shape == (4, 2)
            assert sample.bboxes is not None
            assert sample.labels is not None

        assert counts == {Subset.TRAINING: 2, Subset.VALIDATION: 1, Subset.TESTING: 1}

    def test_missing_train_json_raises(self, tmp_path: Path) -> None:
        (tmp_path / "train").mkdir()
        with pytest.raises(RuntimeError, match="Expected train.json"):
            _build_dataset(tmp_path)


class TestPrepareCarddInstsegRawDir:
    """Exercises the --raw-dir path -- no network, no Kaggle credentials required."""

    def test_prepares_dataset_from_raw_dir(self, tmp_path: Path) -> None:
        assert _SCRIPT.is_file(), f"Script not found: {_SCRIPT}"

        raw_dir = tmp_path / "manually_downloaded_raw"
        _make_synthetic_coco_dataset(raw_dir)

        output_dir = tmp_path / "output"
        result = subprocess.run(
            [
                sys.executable,
                str(_SCRIPT),
                "--output-dir",
                str(output_dir),
                "--name",
                "cardd_instseg",
                "--raw-dir",
                str(raw_dir),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, (
            f"Script failed (exit {result.returncode})\n--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        )

        dataset_dir = output_dir / "cardd_instseg"
        assert dataset_dir.is_dir(), f"Expected output directory not found: {dataset_dir}"

        assert raw_dir.exists()
        assert (raw_dir / "train" / "000001.jpg").exists()

        dataset = import_dataset(dataset_dir)
        assert len(dataset) == 4
        for sample in dataset:
            assert sample.polygons is not None
            assert sample.bboxes is not None

    def test_missing_raw_dir_fails_cleanly(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "output"
        result = subprocess.run(
            [
                sys.executable,
                str(_SCRIPT),
                "--output-dir",
                str(output_dir),
                "--name",
                "cardd_instseg",
                "--raw-dir",
                str(tmp_path / "does_not_exist"),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode != 0
        assert "does not exist" in result.stderr


@pytest.mark.network
@pytest.mark.skipif(
    not _has_kaggle_credentials(),
    reason="Kaggle API token not configured (set KAGGLE_API_TOKEN or ~/.kaggle/access_token)",
)
class TestPrepareCarddInstsegKaggleDownload:
    def test_prepare_cardd_end_to_end(self, tmp_path: Path) -> None:
        assert _SCRIPT.is_file(), f"Script not found: {_SCRIPT}"

        result = subprocess.run(
            [sys.executable, str(_SCRIPT), "--output-dir", str(tmp_path), "--name", "cardd_instseg"],
            check=False,
            capture_output=True,
            text=True,
            timeout=1200,
        )
        assert result.returncode == 0, (
            f"Script failed (exit {result.returncode})\n--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        )

        dataset_dir = tmp_path / "cardd_instseg"
        assert dataset_dir.is_dir(), f"Expected output directory not found: {dataset_dir}"

        dataset = import_dataset(dataset_dir)
        assert len(dataset) > 0
        sample = next(iter(dataset))
        assert sample.polygons is not None
