# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import torch
from datumaro import Polygon
from otx.algo.explain.explain_algo import (
    ActivationMap,
    DetClassProbabilityMap,
    InstSegExplainAlgo,
    ReciproCAM,
    ViTReciproCAM,
)
from otx.core.data.entity.base import ImageInfo
from otx.core.data.entity.instance_segmentation import InstanceSegBatchPredEntity
from torch import LongTensor
from torchvision import tv_tensors


def test_activationmap() -> None:
    explain_algo = ActivationMap()

    assert explain_algo._norm_saliency_maps

    feature_map = torch.zeros((1, 10, 5, 5))

    saliency_maps = explain_algo.func(feature_map)
    assert saliency_maps.size() == torch.Size([1, 5, 5])


def test_reciprocam() -> None:
    def cls_head_forward_fn(_) -> None:
        return torch.zeros((25, 2))

    num_classes = 2
    optimize_gap = False
    explain_algo = ReciproCAM(
        cls_head_forward_fn,
        num_classes=num_classes,
        optimize_gap=optimize_gap,
    )

    assert explain_algo._norm_saliency_maps

    feature_map = torch.zeros((1, 10, 5, 5))

    saliency_maps = explain_algo.func(feature_map)
    assert saliency_maps.size() == torch.Size([1, 2, 5, 5])


def test_vitreciprocam() -> None:
    def cls_head_forward_fn(_) -> None:
        return torch.zeros((196, 2))

    num_classes = 2
    explain_algo = ViTReciproCAM(
        cls_head_forward_fn,
        num_classes=num_classes,
    )

    assert explain_algo._norm_saliency_maps

    feature_map = torch.zeros((1, 197, 192))

    saliency_maps = explain_algo.func(feature_map)
    assert saliency_maps.size() == torch.Size([1, 2, 14, 14])


def test_detclassprob() -> None:
    num_classes = 2
    num_anchors = [1] * 10
    explain_algo = DetClassProbabilityMap(
        num_classes=num_classes,
        num_anchors=num_anchors,
    )

    assert explain_algo._norm_saliency_maps

    backbone_out = torch.zeros((1, 5, 2, 2, 2))

    saliency_maps = explain_algo.func(backbone_out)
    assert saliency_maps.size() == torch.Size([5, 2, 2, 2])


def test_instseg() -> None:
    num_classes = 2
    explain_algo = InstSegExplainAlgo(
        num_classes=num_classes,
    )

    assert explain_algo._norm_saliency_maps

    # One image, 3 masks to aggregate
    pred = InstanceSegBatchPredEntity(
        batch_size=1,
        masks=tv_tensors.Mask(torch.ones(3, 10, 10)),
        scores=LongTensor([0.1, 0.2, 0.3]),
        labels=LongTensor([0, 0, 1]),
        # not used during saliency map calculation
        images=[tv_tensors.Image(torch.randn(3, 10, 10))],
        imgs_info=[ImageInfo(img_idx=0, img_shape=(10, 10), ori_shape=(10, 10))],
        bboxes=[
            3
            * tv_tensors.BoundingBoxes(
                data=torch.Tensor([0, 0, 5, 5]),
                format="xywh",
                canvas_size=(10, 10),
            ),
        ],
        polygons=[Polygon(points=[1, 1, 2, 2, 3, 3, 4, 4])],
    )

    # 2 images
    saliency_maps = explain_algo.func([pred, pred])
    assert len(saliency_maps) == 2
    assert saliency_maps[0].shape == (2, 10, 10)
