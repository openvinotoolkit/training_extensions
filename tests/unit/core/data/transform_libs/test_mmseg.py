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


@pytest.fixture()
def data_entity() -> tuple[tuple, SegDataEntity]:
    img_size = (224, 224)
    data_entity = SegDataEntity(torch.rand(img_size),
                                ImageInfo(0, img_size[0], img_size[0],
                                          img_size[0], img_size[0]),
                                torch.rand(img_size))
    return img_size, data_entity

class TestLoadAnnotations:
    @pytest.fixture()
    def annotations_loader(self) -> LoadAnnotations:
        return LoadAnnotations()

    def test_transform(self, data_entity, annotations_loader) -> None:
        with pytest.raises(RuntimeError):
            new_results = annotations_loader.transform({})
        results = {"__otx__": data_entity[1]}
        new_results = annotations_loader.transform(results)
        assert isinstance(new_results, dict)
        assert "seg_fields" in new_results
        assert "gt_seg_map" in new_results["seg_fields"]
        assert new_results["gt_seg_map"].shape == data_entity[0]


class TestPackSegInputs:
    @pytest.fixture()
    def pack_inputs(self) -> None:
        return PackSegInputs()

    def test_transform(self, mocker, pack_inputs, data_entity) -> None:
        results = {"__otx__": data_entity[1], "gt_seg_map": np.random.rand(*data_entity[0])} # noqa: NPY002
        mocker.patch.object(MMSegPackInputs, "transform", return_value={"inputs": torch.rand(data_entity[0]),
                                                                        "data_samples": MagicMock()})
        output = pack_inputs.transform(results)
        assert isinstance(output, SegDataEntity)
