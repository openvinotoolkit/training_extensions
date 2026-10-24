# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Checks that persisted dataset fixtures follow the current sample label schema."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
import pytest
from datumaro.experimental.export_import import import_dataset

from getitune.data.dataset.base import VisionDataset
from getitune.data.dataset.classification import MulticlassClsDataset
from getitune.data.dataset.detection import DetectionDataset
from getitune.data.entity.sample import (
    ClassificationMultiLabelSample,
    ClassificationSample,
    DetectionSample,
    InstanceSegmentationSample,
    KeypointSample,
)

if TYPE_CHECKING:
    from datumaro.experimental.dataset import Sample

ASSETS_DIR = Path(__file__).resolve().parents[3] / "assets"
FIXTURE_SAMPLES: dict[str, type[Sample]] = {
    "classification_cifar10": ClassificationSample,
    "classification_dataset_16bit": ClassificationSample,
    "detection_coco": DetectionSample,
    "instance_segmentation_coco": InstanceSegmentationSample,
    "keypoint_detection_coco": KeypointSample,
    "multilabel_classification_coco": ClassificationMultiLabelSample,
}
_LABEL_DTYPE_RE = re.compile(r'("type": "LabelField",\s*"semantic": "[^"]*",\s*"dtype": )"UInt16"')


def _label_inner_dtype(df: pl.DataFrame) -> pl.DataType:
    dtype = df["label"].dtype
    return dtype.inner if isinstance(dtype, pl.List) else dtype  # type: ignore[return-value]


def _write_legacy_uint8_copy(fixture_dir: Path, target_dir: Path) -> None:
    """Copy a fixture and rewrite its label schema and stored column to the previous ``UInt8`` format."""
    target_dir.mkdir()
    metadata, count = _LABEL_DTYPE_RE.subn(r'\1"UInt8"', (fixture_dir / "metadata.json").read_text())
    assert count == 1
    (target_dir / "metadata.json").write_text(metadata)
    fixture_df = pl.read_parquet(fixture_dir / "data.parquet")
    legacy_dtype = pl.List(pl.UInt8()) if isinstance(fixture_df["label"].dtype, pl.List) else pl.UInt8()
    fixture_df.with_columns(pl.col("label").cast(legacy_dtype, strict=True)).write_parquet(target_dir / "data.parquet")
    for extra in fixture_dir.iterdir():
        if extra.is_dir():
            shutil.copytree(extra, target_dir / extra.name)


@pytest.mark.parametrize(("fixture_name", "sample_type"), list(FIXTURE_SAMPLES.items()))
def test_persisted_label_schema_matches_sample(fixture_name: str, sample_type: type[Sample]) -> None:
    """Persisted fixtures must declare and store labels with the current sample schema (UInt16)."""
    fixture_dir = ASSETS_DIR / fixture_name
    stored = json.loads((fixture_dir / "metadata.json").read_text())["schema"]["attributes"]["label"]["field"]
    expected = sample_type.infer_schema().to_dict()["attributes"]["label"]["field"]
    assert stored == expected
    assert _label_inner_dtype(pl.read_parquet(fixture_dir / "data.parquet")) == pl.UInt16()


@pytest.mark.parametrize(
    ("fixture_name", "dataset_cls"),
    [("classification_cifar10", MulticlassClsDataset), ("detection_coco", DetectionDataset)],
)
def test_legacy_uint8_persisted_labels_are_widened(
    tmp_path: Path, fixture_name: str, dataset_cls: type[VisionDataset]
) -> None:
    """Datasets persisted with the previous UInt8 label schema still load with identical labels."""
    fixture_dir = ASSETS_DIR / fixture_name
    legacy_dir = tmp_path / fixture_name
    _write_legacy_uint8_copy(fixture_dir, legacy_dir)
    assert _label_inner_dtype(pl.read_parquet(legacy_dir / "data.parquet")) == pl.UInt8()
    current = dataset_cls(dm_subset=import_dataset(fixture_dir), transforms=[])  # type: ignore[call-arg]
    legacy = dataset_cls(dm_subset=import_dataset(legacy_dir), transforms=[])  # type: ignore[call-arg]
    assert len(legacy) == len(current)
    assert legacy.get_idx_list_per_classes() == current.get_idx_list_per_classes()  # type: ignore[attr-defined]
