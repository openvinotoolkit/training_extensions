# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``scripts/benchmark_datasets/prepare_food41.py``."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from datumaro.experimental.fields import Subset
from PIL import Image
from scripts.benchmark_datasets.prepare_food41 import _build_dataset, _find_food_root, _resolve_split_file


def _make_food_root(root: Path) -> Path:
    data_root = root / "food-101"
    (data_root / "images").mkdir(parents=True)
    (data_root / "meta").mkdir(parents=True)

    train = ["apple_pie/00000001", "apple_pie/00000002", "beef_carpaccio/00000001", "beef_carpaccio/00000002"]
    test = ["apple_pie/00000003", "beef_carpaccio/00000003"]
    (data_root / "meta" / "train.txt").write_text("\n".join(train) + "\n", encoding="utf-8")
    (data_root / "meta" / "test.txt").write_text("\n".join(test) + "\n", encoding="utf-8")

    for item in train + test:
        class_name, image_name = item.split("/")
        img_dir = data_root / "images" / class_name
        img_dir.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8), color=(0, 0, 0)).save(img_dir / f"{image_name}.jpg")

    return root


def _make_direct_food_root(root: Path) -> Path:
    (root / "images").mkdir(parents=True)
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "train.txt").write_text("apple_pie/00000001\n", encoding="utf-8")
    (root / "meta" / "test.txt").write_text("apple_pie/00000002\n", encoding="utf-8")
    img_dir = root / "images" / "apple_pie"
    img_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=(0, 0, 0)).save(img_dir / "00000001.jpg")
    Image.new("RGB", (8, 8), color=(0, 0, 0)).save(img_dir / "00000002.jpg")
    return root


def _make_nested_meta_food_root(root: Path) -> Path:
    (root / "images").mkdir(parents=True)
    (root / "meta" / "meta").mkdir(parents=True)
    (root / "meta" / "meta" / "train.txt").write_text("apple_pie/00000001\n", encoding="utf-8")
    (root / "meta" / "meta" / "test.txt").write_text("apple_pie/00000002\n", encoding="utf-8")
    img_dir = root / "images" / "apple_pie"
    img_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=(0, 0, 0)).save(img_dir / "00000001.jpg")
    Image.new("RGB", (8, 8), color=(0, 0, 0)).save(img_dir / "00000002.jpg")
    return root


def test_find_food_root(tmp_path: Path) -> None:
    root = _make_food_root(tmp_path)
    assert _find_food_root(root) == root / "food-101"


def test_find_food_root_direct_layout(tmp_path: Path) -> None:
    root = _make_direct_food_root(tmp_path)
    assert _find_food_root(root) == root


def test_find_food_root_nested_meta_layout(tmp_path: Path) -> None:
    root = _make_nested_meta_food_root(tmp_path)
    assert _find_food_root(root) == root


def test_resolve_split_file_nested_meta_layout(tmp_path: Path) -> None:
    root = _make_nested_meta_food_root(tmp_path)
    assert _resolve_split_file(root, "train") == root / "meta" / "meta" / "train.txt"


def test_build_dataset(tmp_path: Path) -> None:
    root = _make_food_root(tmp_path)
    dataset = _build_dataset(root / "food-101")

    assert len(dataset) == 6
    subsets = [sample.subset for sample in dataset]
    assert subsets.count(Subset.TRAINING) == 2
    assert subsets.count(Subset.VALIDATION) == 2
    assert subsets.count(Subset.TESTING) == 2


def test_missing_raw_dir_fails_cleanly(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[3] / "scripts" / "benchmark_datasets" / "prepare_food41.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--output-dir",
            str(tmp_path / "out"),
            "--name",
            "food41",
            "--raw-dir",
            str(tmp_path / "missing"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "does not exist" in result.stderr
