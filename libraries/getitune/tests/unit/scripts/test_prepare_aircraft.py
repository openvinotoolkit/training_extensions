# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``scripts/benchmark_datasets/prepare_aircraft.py``."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from datumaro.experimental.fields import Subset
from PIL import Image
from scripts.benchmark_datasets.prepare_aircraft import _build_dataset, _find_data_root


def _make_fgvc_root(root: Path) -> Path:
    data_root = root / "fgvc-aircraft-2013b" / "data"
    (data_root / "images").mkdir(parents=True)

    labels = ["Boeing 737-700", "Airbus A320-200"]
    for split in ("train", "val", "test"):
        split_lines = []
        for idx, label in enumerate(labels):
            image_name = f"{split}_{idx:07d}"
            Image.new("RGB", (16, 16), color=(idx * 40, 0, 0)).save(data_root / "images" / f"{image_name}.jpg")
            split_lines.append(f"{image_name} {label}\n")
        (data_root / f"images_variant_{split}.txt").write_text("".join(split_lines), encoding="utf-8")

    return root


def test_find_data_root(tmp_path: Path) -> None:
    root = _make_fgvc_root(tmp_path)
    assert _find_data_root(root) == root / "fgvc-aircraft-2013b" / "data"


def test_build_dataset(tmp_path: Path) -> None:
    root = _make_fgvc_root(tmp_path)
    dataset = _build_dataset(root / "fgvc-aircraft-2013b" / "data", annotation_level="variant")

    assert len(dataset) == 6
    subsets = [sample.subset for sample in dataset]
    assert subsets.count(Subset.TRAINING) == 2
    assert subsets.count(Subset.VALIDATION) == 2
    assert subsets.count(Subset.TESTING) == 2


def test_missing_raw_dir_fails_cleanly(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[3] / "scripts" / "benchmark_datasets" / "prepare_aircraft.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--output-dir",
            str(tmp_path / "out"),
            "--name",
            "aircraft",
            "--raw-dir",
            str(tmp_path / "missing"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "does not exist" in result.stderr
