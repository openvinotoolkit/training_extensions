# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Helper to support MMPretrain data transform functions."""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from otx.core.data.entity.base import ImageInfo
from otx.core.data.entity.segmentation import SegDataEntity
from otx.core.data.transform_libs.mmseg import LoadAnnotations, MMSegPackInputs, PackSegInputs


class TestLoadAnnotations:
    @pytest.fixture()
    def annotations_loader(self) -> LoadAnnotations:
        return LoadAnnotations()

    def test_transform(self, fxt_seg_data_entity, annotations_loader) -> None:
        with pytest.raises(RuntimeError):
            new_results = annotations_loader.transform({})
        results = {"__otx__": fxt_seg_data_entity[0]}
        new_results = annotations_loader.transform(results)
        assert isinstance(new_results, dict)
        assert "seg_fields" in new_results
        assert "gt_seg_map" in new_results["seg_fields"]
        assert new_results["gt_seg_map"].size == fxt_seg_data_entity[0].img_info.img_shape**2


class TestPackSegInputs:
    @pytest.fixture()
    def pack_inputs(self) -> None:
        return PackSegInputs()

    def test_transform(self, mocker, pack_inputs, fxt_seg_data_entity) -> None:
        img_size = fxt_seg_data_entity[0].img_info.img_shape
        results = {"__otx__": fxt_seg_data_entity[0], "gt_seg_map": np.random.rand(img_size, img_size)} # noqa: NPY002
        mocker.patch.object(MMSegPackInputs, "transform", return_value={"inputs": torch.rand((img_size, img_size)),
                                                                        "data_samples": MagicMock()})
        output = pack_inputs.transform(results)
        assert isinstance(output, SegDataEntity)
