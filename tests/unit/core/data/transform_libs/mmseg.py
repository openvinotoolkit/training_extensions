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
def common_fxt() -> tuple[tuple, SegDataEntity]:
    img_size = (224, 224)
    data_entity = SegDataEntity(torch.rand(img_size),
                                         ImageInfo(0, img_size[0], img_size[0],
                                                   img_size[0], img_size[0]),
                                         torch.rand(img_size))
    return img_size, data_entity

class TestLoadAnnotations:
    @pytest.fixture(autouse=True)
    def setup(self, common_fxt) -> None:
        self.load_ann = LoadAnnotations()
        _, self.data_entity = common_fxt

    def test_transform(self) -> None:
        with pytest.raises(RuntimeError):
            new_results = self.load_ann.transform({})
        results = {"__otx__": self.data_entity}
        new_results = self.load_ann.transform(results)
        assert isinstance(new_results, dict)
        assert "seg_fields" in new_results
        assert "gt_seg_map" in new_results["seg_fields"]
        assert new_results["gt_seg_map"].shape == (224,224)


class TestPackSegInputs:
    @pytest.fixture(autouse=True)
    def setup(self, common_fxt) -> None:
        self.pack_inputs = PackSegInputs()
        self.img_size, self.data_entity = common_fxt

    def test_transform(self, mocker) -> None:
        results = {"__otx__": self.data_entity, "gt_seg_map": np.random.rand(*self.img_size)} # noqa: NPY002
        mocker.patch.object(MMSegPackInputs, "transform", return_value={"inputs": torch.rand(self.img_size),
                                                                        "data_samples": MagicMock()})
        output = self.pack_inputs.transform(results)
        assert isinstance(output, SegDataEntity)
