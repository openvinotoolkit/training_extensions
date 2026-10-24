# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for utils for getitune data module."""

from __future__ import annotations

from unittest.mock import MagicMock

import polars as pl
import pytest

from getitune.data.utils import utils as target_file
from getitune.data.utils.utils import (
    _is_simple_scalar_list,
    fill_null_annotation_lists,
    get_adaptive_num_workers,
)


@pytest.mark.parametrize("num_dataloader", [1, 2, 4])
def test_get_adaptive_num_workers(mocker, num_dataloader):
    num_gpu = 5
    mocker.patch.object(target_file, "is_xpu_available", return_value=False)
    mock_torch = mocker.patch.object(target_file, "torch")
    mock_torch.cuda.device_count.return_value = num_gpu

    num_cpu = 20
    mocker.patch.object(target_file, "cpu_count", return_value=num_cpu)

    assert get_adaptive_num_workers(num_dataloader) == min(num_cpu // (num_gpu * num_dataloader), 8)


def test_get_adaptive_num_workers_no_gpu(mocker):
    num_gpu = 0
    mocker.patch.object(target_file, "is_xpu_available", return_value=False)
    mock_torch = mocker.patch.object(target_file, "torch")
    mock_torch.cuda.device_count.return_value = num_gpu

    num_cpu = 20
    mocker.patch.object(target_file, "cpu_count", return_value=num_cpu)

    assert get_adaptive_num_workers() is None


class TestIsSimpleScalarList:
    """Unit tests for `_is_simple_scalar_list`."""

    @pytest.mark.parametrize(
        ("dtype", "expected"),
        [
            (pl.List(pl.UInt32), True),
            (pl.List(pl.Float32), True),
            (pl.List(pl.Boolean), True),
            (pl.List(pl.String), True),
            (pl.List(pl.Array(pl.Float32, shape=(4,))), False),
            (pl.List(pl.Array(pl.Float32, shape=(3,))), False),
            (pl.List(pl.Struct({"a": pl.Int32()})), False),
            # Nested lists (e.g. polygons: List(List(Array))) must unwrap all
            # levels of nesting before checking the innermost element type.
            (pl.List(pl.List(pl.Array(pl.Float32, shape=(2,)))), False),
            (pl.List(pl.List(pl.List(pl.UInt32))), True),
        ],
    )
    def test_is_simple_scalar_list(self, dtype: pl.List, expected: bool) -> None:
        assert _is_simple_scalar_list(dtype) is expected


class TestFillNullAnnotationLists:
    """Unit tests for `fill_null_annotation_lists`."""

    def test_fills_simple_scalar_null_columns(self) -> None:
        dataframe = pl.DataFrame(
            {
                "labels": [[1, 2], None, [3]],
                "areas": [[1.0], None, [2.0, 3.0]],
            },
            schema={"labels": pl.List(pl.UInt32), "areas": pl.List(pl.Float32)},
        )
        dm_dataset = MagicMock(df=dataframe)

        result = fill_null_annotation_lists(dm_dataset)

        assert result.df["labels"].null_count() == 0
        assert result.df["labels"][1].to_list() == []
        assert result.df["areas"].null_count() == 0
        assert result.df["areas"][1].to_list() == []

    def test_does_not_fill_array_backed_list_columns(self) -> None:
        """Columns like bboxes/polygons must be left untouched (nullable) to avoid dtype corruption."""
        dataframe = pl.DataFrame(
            {
                "bboxes": [[[0.0, 0.0, 1.0, 1.0]], None],
                "polygons": [[[[0.0, 0.0], [1.0, 1.0]]], None],
            },
            schema={
                "bboxes": pl.List(pl.Array(pl.Float32, shape=(4,))),
                "polygons": pl.List(pl.List(pl.Array(pl.Float32, shape=(2,)))),
            },
        )
        dm_dataset = MagicMock(df=dataframe)

        result = fill_null_annotation_lists(dm_dataset)

        assert result.df["bboxes"].null_count() == 1
        assert result.df["polygons"].null_count() == 1

    def test_leaves_non_list_columns_untouched(self) -> None:
        dataframe = pl.DataFrame({"image_id": [1, 2, 3]}, schema={"image_id": pl.Int32})
        dm_dataset = MagicMock(df=dataframe)

        result = fill_null_annotation_lists(dm_dataset)

        assert result.df["image_id"].to_list() == [1, 2, 3]

    @pytest.mark.parametrize("bad_dataset", [MagicMock(df=None), object(), "not-a-dataset", 42])
    def test_defensive_on_non_dataframe_like_input(self, bad_dataset) -> None:
        """Should not raise for dataset-like test doubles without a real dataframe."""
        result = fill_null_annotation_lists(bad_dataset)
        assert result is bad_dataset
