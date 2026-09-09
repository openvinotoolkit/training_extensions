# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``scripts/benchmark_datasets/prepare_brain_tumor.py``.

Two test tiers:

1. The ``--raw-dir`` path is exercised unconditionally (no network, no Kaggle
   credentials needed) against a small synthetic Roboflow-style YOLO-segmentation
   export -- proving that this credentialed dataset can be prepared entirely
   offline once its raw data has been fetched once (e.g. manually, or via a
   prior Kaggle-authenticated run). This is what runs on every PR.
2. The real Kaggle-download path is an end-to-end ``@pytest.mark.network``
   test, additionally skipped when Kaggle credentials aren't configured in
   the current environment -- so forked-repo CI runs (which never receive
   GitHub Actions secrets) skip it instead of failing.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from datumaro.experimental.export_import import import_dataset
from PIL import Image
from scripts.benchmark_datasets.prepare_brain_tumor import (
    _find_yolo_root,
    _load_label_names,
    _parse_yolo_segmentation_label,
)

from getitune.benchmark.dataset_helpers import _has_kaggle_credentials

_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "benchmark_datasets" / "prepare_brain_tumor.py"

# A simple square polygon (4 points) in normalized [0, 1] coordinates.
_POLYGON_LINE = "0 0.25 0.25 0.75 0.25 0.75 0.75 0.25 0.75\n"


def _make_synthetic_yolo_dir(root: Path, n_images: int = 3) -> None:
    """Create a tiny Roboflow-style ("subset-first") YOLO-segmentation export.

    Layout::

        root/data.yaml
        root/train/images/sample_0.png ...
        root/train/labels/sample_0.txt ...
    """
    images_dir = root / "train" / "images"
    labels_dir = root / "train" / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    root.joinpath("data.yaml").write_text(
        "train: ../train/images\nval: ../valid/images\ntest: ../test/images\nnc: 1\nnames: ['tumor']\n",
    )
    for i in range(n_images):
        Image.new("RGB", (8, 8), color=(i * 10, 0, 0)).save(images_dir / f"sample_{i}.png")
        (labels_dir / f"sample_{i}.txt").write_text(_POLYGON_LINE)


class TestFindYoloRoot:
    def test_finds_roboflow_yolo_root(self, tmp_path: Path) -> None:
        nested = tmp_path / "outer" / "inner"
        _make_synthetic_yolo_dir(nested)
        assert _find_yolo_root(tmp_path) == nested

    def test_missing_root_raises(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="No Roboflow-style YOLO dataset root found"):
            _find_yolo_root(tmp_path)


class TestLoadLabelNames:
    def test_reads_names_list(self, tmp_path: Path) -> None:
        _make_synthetic_yolo_dir(tmp_path)
        assert _load_label_names(tmp_path) == ("tumor",)

    def test_reads_names_dict(self, tmp_path: Path) -> None:
        _make_synthetic_yolo_dir(tmp_path)
        tmp_path.joinpath("data.yaml").write_text("nc: 2\nnames: {0: tumor, 1: other}\n")
        assert _load_label_names(tmp_path) == ("tumor", "other")


class TestParseYoloSegmentationLabel:
    def test_missing_label_file_returns_all_none(self, tmp_path: Path) -> None:
        result = _parse_yolo_segmentation_label(tmp_path / "missing.txt", width=100, height=100)
        assert result == (None, None, None, None, None)

    def test_parses_polygon_line(self, tmp_path: Path) -> None:
        label_path = tmp_path / "sample.txt"
        label_path.write_text(_POLYGON_LINE)

        bboxes, labels, areas, iscrowd, polygons = _parse_yolo_segmentation_label(label_path, width=100, height=100)

        assert labels is not None
        assert bboxes is not None
        assert areas is not None
        assert iscrowd is not None
        assert polygons is not None
        assert labels.tolist() == [0]
        # Polygon spans [0.25, 0.75] normalized -> [25, 75] px in a 100x100 image.
        assert bboxes[0].tolist() == pytest.approx([25.0, 25.0, 50.0, 50.0])
        assert areas[0] == pytest.approx(50.0 * 50.0)
        assert iscrowd.tolist() == [False]
        assert polygons[0].shape == (4, 2)

    def test_skips_malformed_lines(self, tmp_path: Path) -> None:
        label_path = tmp_path / "sample.txt"
        # Even token count (bbox-style, not polygon) should be skipped.
        label_path.write_text("0 0.5 0.5 0.2 0.2\n")

        result = _parse_yolo_segmentation_label(label_path, width=100, height=100)

        assert result == (None, None, None, None, None)


class TestPrepareBrainTumorRawDir:
    """Exercises the --raw-dir path — no network, no Kaggle credentials required."""

    def test_prepares_dataset_from_raw_dir(self, tmp_path: Path) -> None:
        assert _SCRIPT.is_file(), f"Script not found: {_SCRIPT}"

        raw_dir = tmp_path / "manually_downloaded_raw"
        _make_synthetic_yolo_dir(raw_dir, n_images=3)

        output_dir = tmp_path / "output"
        result = subprocess.run(
            [
                sys.executable,
                str(_SCRIPT),
                "--output-dir",
                str(output_dir),
                "--name",
                "brain_tumor",
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

        dataset_dir = output_dir / "brain_tumor"
        assert dataset_dir.is_dir(), f"Expected output directory not found: {dataset_dir}"

        # A manually-supplied --raw-dir is externally owned and must never be deleted.
        assert raw_dir.exists()
        assert (raw_dir / "train" / "images" / "sample_0.png").exists()

        dataset = import_dataset(dataset_dir)
        assert len(dataset) == 3
        for sample in dataset:
            assert sample.labels.tolist() == [0]
            assert sample.polygons[0].shape == (4, 2)

    def test_missing_raw_dir_fails_cleanly(self, tmp_path: Path) -> None:
        """A --raw-dir that doesn't exist should fail with a clear, actionable error."""
        output_dir = tmp_path / "output"
        result = subprocess.run(
            [
                sys.executable,
                str(_SCRIPT),
                "--output-dir",
                str(output_dir),
                "--name",
                "brain_tumor",
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
class TestPrepareBrainTumorKaggleDownload:
    """Real Kaggle download — requires --run-network and configured credentials.

    In CI this only exercises on same-repo runs where the KAGGLE_API_TOKEN secret
    is available (see .github/workflows/lib-lint-and-test.yaml);
    forked-repo PRs skip it via the `skipif` above rather than failing.
    """

    def test_prepare_brain_tumor_end_to_end(self, tmp_path: Path) -> None:
        assert _SCRIPT.is_file(), f"Script not found: {_SCRIPT}"

        result = subprocess.run(
            [sys.executable, str(_SCRIPT), "--output-dir", str(tmp_path), "--name", "brain_tumor"],
            check=False,
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert result.returncode == 0, (
            f"Script failed (exit {result.returncode})\n--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        )

        dataset_dir = tmp_path / "brain_tumor"
        assert dataset_dir.is_dir(), f"Expected output directory not found: {dataset_dir}"

        dataset = import_dataset(dataset_dir)
        assert len(dataset) > 0
