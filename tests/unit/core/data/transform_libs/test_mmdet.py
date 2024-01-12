# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Unit tests of mmdet transforms."""

import numpy as np
import torch
from otx.core.config.data import SubsetConfig
from otx.core.data.entity.base import ImageInfo
from otx.core.data.entity.detection import DetDataEntity
from otx.core.data.transform_libs.mmcv import LoadImageFromFile
from otx.core.data.transform_libs.mmdet import LoadAnnotations, MMDetTransformLib, PackDetInputs
from otx.core.types.transformer_libs import TransformLibType
from torch import LongTensor
from torchvision import tv_tensors


class TestLoadAnnotations:
    def test_transform(self) -> None:
        data_entity = DetDataEntity(
            tv_tensors.Image(torch.randn(3, 224, 224)),
            ImageInfo(img_idx=0, img_shape=(224, 224), ori_shape=(224, 224), attributes={"idx": 0}),
            tv_tensors.BoundingBoxes(data=torch.Tensor([0, 0, 50, 50]), format="xywh", canvas_size=(224, 224)),
            LongTensor([1]),
        )

        data_entity = LoadImageFromFile().transform(data_entity)
        transform = LoadAnnotations()
        results = transform.transform(data_entity)
        assert np.all(results["gt_bboxes"] == np.array([[0.0, 0.0, 50.0, 50.0]]))
        assert results["gt_bboxes_labels"] == np.array([1])
        assert results["gt_ignore_flags"] == np.array([False])


class TestPackDetInputs:
    def test_transform(self) -> None:
        transform = PackDetInputs()
        data_entity = DetDataEntity(
            image=np.ndarray((224, 224, 3)),
            img_info=ImageInfo(img_idx=0, img_shape=(224, 224), ori_shape=(224, 224), attributes={"idx": 0}),
            bboxes=tv_tensors.BoundingBoxes(data=torch.Tensor([0, 0, 50, 50]), format="xywh", canvas_size=(224, 224)),
            labels=LongTensor([1]),
        )
        data_entity = LoadAnnotations().transform(LoadImageFromFile().transform(data_entity))
        results = transform.transform(data_entity)
        assert results.image.shape == torch.Size([3, 224, 224])


class TestMMDetTransformLib:
    def test_generate(self, mocker) -> None:
        def mock_convert_func(cfg: dict) -> dict:
            return cfg

        mocker.patch("otx.core.data.transform_libs.mmcv.convert_conf_to_mmconfig_dict", side_effect=mock_convert_func)
        config = SubsetConfig(
            batch_size=64,
            subset_name="train",
            transform_lib_type=TransformLibType.MMCV,
            transforms=[
                {"type": "LoadImageFromFile"},
                {"type": "LoadAnnotations"},
                {"type": "Normalize", "mean": [0, 0, 0], "std": [1, 1, 1]},
                {"type": "PackDetInputs"},
            ],
            num_workers=2,
        )

        transforms = MMDetTransformLib.generate(config)
        assert len(transforms) == 4
