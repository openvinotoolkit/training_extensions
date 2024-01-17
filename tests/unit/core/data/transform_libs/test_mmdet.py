# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Unit tests of mmdet transforms."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from otx.core.config.data import SubsetConfig
from otx.core.data.entity.base import ImageInfo
from otx.core.data.entity.detection import DetDataEntity
from otx.core.data.entity.visual_prompting import VisualPromptingDataEntity
from otx.core.data.transform_libs.mmcv import LoadImageFromFile
from otx.core.data.transform_libs.mmdet import LoadAnnotations, MMDetTransformLib, PackDetInputs, PerturbBoundingBoxes
from otx.core.types.transformer_libs import TransformLibType
from torch import LongTensor
from torchvision import tv_tensors


class TestLoadAnnotations:
    def test_transform(self) -> None:
        data_entity = DetDataEntity(
            tv_tensors.Image(torch.randn(3, 224, 224)),
            ImageInfo(img_idx=0, img_shape=(224, 224), ori_shape=(224, 224)),
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
    @pytest.mark.parametrize(
        ("data_entity", "expected"),
        [
            (
                DetDataEntity(
                    image=np.ndarray((224, 224, 3)),
                    img_info=ImageInfo(img_idx=0, img_shape=(224, 224), ori_shape=(224, 224)),
                    bboxes=tv_tensors.BoundingBoxes(
                        data=torch.Tensor([0, 0, 50, 50]),
                        format="xywh",
                        canvas_size=(224, 224),
                    ),
                    labels=LongTensor([1]),
                ),
                torch.Size([3, 224, 224]),
            ),
            (
                VisualPromptingDataEntity(
                    image=np.ndarray((1024, 1024, 3)),
                    img_info=ImageInfo(img_idx=0, img_shape=(1024, 1024), ori_shape=(1024, 1024)),
                    bboxes=tv_tensors.BoundingBoxes(
                        data=torch.Tensor([0, 0, 50, 50]),
                        format=tv_tensors.BoundingBoxFormat.XYXY,
                        canvas_size=(1024, 1024),
                    ),
                    masks=None,
                    labels=LongTensor([1]),
                    polygons=None,
                ),
                torch.Size([3, 1024, 1024]),
            ),
        ],
    )
    def test_transform(self, data_entity: DetDataEntity | VisualPromptingDataEntity, expected: torch.Size) -> None:
        transform = PackDetInputs()
        data_entity = LoadAnnotations().transform(LoadImageFromFile().transform(data_entity))

        results = transform.transform(data_entity)

        assert results.image.shape == expected


class TestPerturbBoundingBoxes:
    def test_transform(self) -> None:
        transform = PerturbBoundingBoxes(offset=20)
        inputs = {
            "img_shape": (300, 300),
            "gt_bboxes": np.array(
                [
                    [100, 100, 200, 200],  # normal bbox
                    [0, 0, 299, 299],  # can be out of size
                    [0, 0, 100, 100],  # can be out of size
                    [100, 100, 299, 299],  # can be out of size
                ],
            ),
        }

        results = transform.transform(inputs)

        assert isinstance(results, dict)
        assert results["gt_bboxes"].shape == inputs["gt_bboxes"].shape
        assert np.all(results["gt_bboxes"][1] == inputs["gt_bboxes"][1])  # check if clipped
        assert np.all(results["gt_bboxes"][2][:2] == inputs["gt_bboxes"][2][:2])
        assert np.all(results["gt_bboxes"][3][2:] == inputs["gt_bboxes"][3][2:])


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
