# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Unit tests of mmdet transforms."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from datumaro import Polygon
from otx.core.config.data import SubsetConfig
from otx.core.data.entity.base import ImageInfo
from otx.core.data.entity.detection import DetDataEntity
from otx.core.data.entity.instance_segmentation import InstanceSegDataEntity
from otx.core.data.entity.visual_prompting import VisualPromptingDataEntity
from otx.core.types.transformer_libs import TransformLibType
from torch import LongTensor
from torchvision import tv_tensors

SKIP_MMLAB_TEST = False
try:
    from mmdet.structures.mask import PolygonMasks
    from otx.core.data.transform_libs.mmcv import LoadImageFromFile
    from otx.core.data.transform_libs.mmdet import (
        LoadAnnotations,
        MMDetTransformLib,
        PackDetInputs,
    )
except ImportError:
    SKIP_MMLAB_TEST = True


@pytest.mark.skipif(SKIP_MMLAB_TEST, reason="MMLab is not installed")
class TestLoadAnnotations:
    def test_det_transform(self) -> None:
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

    def test_vp_transform(self) -> None:
        data_entity = VisualPromptingDataEntity(
            image=np.ndarray([3, 10, 10]),
            img_info=ImageInfo(
                img_idx=0,
                img_shape=(10, 10),
                ori_shape=(10, 10),
            ),
            masks=[],
            labels=LongTensor([0]),
            polygons=[Polygon(points=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], label=0)],
            points=torch.Tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]),
            bboxes=torch.Tensor([[0, 0, 1, 1]]),
        )
        data_entity = LoadImageFromFile().transform(data_entity)
        transform = LoadAnnotations(with_mask=True)
        results = transform.transform(data_entity)
        assert np.all(results["gt_bboxes"] == np.array([[0.0, 0.0, 1.0, 1.0]]))
        assert isinstance(results["gt_masks"], PolygonMasks)
        assert results["gt_bboxes_labels"] == np.array([0])
        assert results["gt_ignore_flags"] == np.array([False])


@pytest.mark.skipif(SKIP_MMLAB_TEST, reason="MMLab is not installed")
class TestPackDetInputs:
    @pytest.mark.parametrize(
        ("data_entity", "with_point", "expected"),
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
                False,
                torch.Size([3, 224, 224]),
            ),
            (
                InstanceSegDataEntity(
                    image=np.ndarray([10, 10, 3]),
                    img_info=ImageInfo(
                        img_idx=0,
                        img_shape=(10, 10),
                        ori_shape=(10, 10),
                    ),
                    masks=[],
                    labels=LongTensor([0]),
                    polygons=[Polygon(points=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], label=0)],
                    bboxes=torch.Tensor([[0, 0, 1, 1]]),
                ),
                False,
                torch.Size([3, 10, 10]),
            ),
            (
                VisualPromptingDataEntity(
                    image=np.ndarray((1024, 1024, 3)),
                    img_info=ImageInfo(img_idx=0, img_shape=(1024, 1024), ori_shape=(1024, 1024)),
                    bboxes=tv_tensors.BoundingBoxes(
                        data=torch.Tensor([[0, 0, 50, 50]]),
                        format=tv_tensors.BoundingBoxFormat.XYXY,
                        canvas_size=(1024, 1024),
                    ),
                    points=None,  # TODO(sungchul): add point prompts in mmx
                    masks=[],
                    labels=LongTensor([1]),
                    polygons=[Polygon(points=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], label=0)],
                ),
                False,  # TODO(sungchul): add point prompts in mmx
                torch.Size([3, 1024, 1024]),
            ),
        ],
    )
    def test_transform(
        self,
        data_entity: DetDataEntity | VisualPromptingDataEntity,
        with_point: bool,
        expected: torch.Size,
    ) -> None:
        transform = PackDetInputs()
        data_entity = LoadAnnotations(with_point=with_point, with_mask=True).transform(
            LoadImageFromFile().transform(data_entity),
        )

        results = transform.transform(data_entity)

        assert results.image.shape == expected


@pytest.mark.skipif(SKIP_MMLAB_TEST, reason="MMLab is not installed")
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
