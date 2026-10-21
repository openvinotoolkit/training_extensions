# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from datumaro.experimental import Dataset, LazyImage, Sample
from datumaro.experimental.categories import LabelCategories
from datumaro.experimental.export_import import export_dataset
from datumaro.experimental.fields import (
    ImageInfo,
    Subset,
    image_info_field,
    image_path_field,
    label_field,
    subset_field,
)
from PIL import Image

from app.datumaro_converter import SampleMode, import_dataset

_SAMPLE_MODES = [SampleMode.TRAINING, SampleMode.IMPORT_EXPORT]


def _legacy_sample_type(sample_type: type[Sample]) -> type[Sample]:
    """Return an unregistered copy of ``sample_type`` with the ``UInt8`` label used by previous app versions."""
    label_config = sample_type.infer_schema().to_dict()["attributes"]["label"]["field"]
    namespace = {
        "__annotations__": {"label": sample_type.__annotations__["label"]},
        "label": label_field(
            dtype=pl.UInt8(), multi_label=label_config["multi_label"], is_list=label_config["is_list"]
        ),
    }
    return type(f"_Legacy{sample_type.__name__}", (sample_type,), namespace)


def _media_paths(tmp_path: Path, count: int) -> list[str]:
    paths = []
    for idx in range(count):
        path = tmp_path / f"image_{idx}.jpg"
        Image.new("RGB", (200, 100)).save(path)
        paths.append(str(path))
    return paths


def _labels(dataset: Dataset) -> list[list[int]]:
    return [np.atleast_1d(dataset[i].label).tolist() for i in range(len(dataset))]


def _has_label_dtype(dataset: Dataset, expected: pl.DataType) -> bool:
    """Check the storage dtype of the label column, looking through list columns."""
    dtype = dataset.df["label"].dtype
    return (dtype.inner if isinstance(dtype, pl.List) else dtype) == expected


@pytest.mark.parametrize("sample_mode", _SAMPLE_MODES)
def test_import_dataset_upgrades_legacy_uint8_schema(
    tmp_path: Path, fxt_task_name: str, sample_mode: SampleMode, fxt_task_dataset
) -> None:
    """Datasets stored with the old UInt8 label (revisions, exported archives) load as the current sample type."""
    label_indices = [0, 128, 255]
    dataset, expected_labels = fxt_task_dataset(
        fxt_task_name,
        label_indices=label_indices,
        sample_mode=sample_mode,
        num_labels=256,
        media_paths=_media_paths(tmp_path, len(label_indices)),
    )
    current_type = dataset.dtype
    legacy_dataset = dataset.convert_to_schema(_legacy_sample_type(current_type))
    assert _has_label_dtype(legacy_dataset, pl.UInt8())
    export_dataset(legacy_dataset, tmp_path / "exported")
    imported = import_dataset(tmp_path / "exported")
    assert imported.dtype is current_type
    assert _has_label_dtype(imported, pl.UInt16())
    assert _labels(imported) == expected_labels
    categories = imported.schema.attributes["label"].categories
    assert isinstance(categories, LabelCategories)
    assert len(categories.labels) == 256


@pytest.mark.parametrize("sample_mode", _SAMPLE_MODES)
def test_import_dataset_roundtrips_label_index_above_uint8(
    tmp_path: Path, fxt_task_name: str, sample_mode: SampleMode, fxt_task_dataset
) -> None:
    label_indices = [255, 256, 299]
    dataset, expected_labels = fxt_task_dataset(
        fxt_task_name,
        label_indices=label_indices,
        sample_mode=sample_mode,
        media_paths=_media_paths(tmp_path, len(label_indices)),
    )
    export_dataset(dataset, tmp_path / "exported")
    imported = import_dataset(tmp_path / "exported")
    assert imported.dtype is dataset.dtype
    assert _labels(imported) == expected_labels


class _UnknownSample(Sample):
    """Sample with a schema that does not match any application sample. Not registered."""

    image: LazyImage = image_path_field()
    image_info: ImageInfo = image_info_field()
    subset: Subset = subset_field()
    label: int | None = label_field(dtype=pl.UInt8(), is_list=False)


def test_import_dataset_keeps_unknown_schema_unchanged(tmp_path: Path) -> None:
    dataset = Dataset(_UnknownSample, categories={"label": LabelCategories(labels=("cat", "dog"))})
    dataset.append(
        _UnknownSample(
            image=LazyImage(_media_paths(tmp_path, 1)[0]),
            image_info=ImageInfo(width=200, height=100),
            subset=Subset.TRAINING,
            label=1,
        )
    )
    export_dataset(dataset, tmp_path / "exported")
    imported = import_dataset(tmp_path / "exported")
    assert imported.dtype is Sample
    assert imported.df["label"].dtype == pl.UInt8()
