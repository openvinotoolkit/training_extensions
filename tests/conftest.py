# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest
import torch
from datumaro import Polygon
from mmdet.structures import DetDataSample
from mmengine.structures import InstanceData
from otx.core.data.entity.base import ImageInfo
from otx.core.data.entity.detection import DetBatchDataEntity, DetBatchPredEntity, DetDataEntity
from otx.core.data.entity.instance_segmentation import (
    InstanceSegBatchDataEntity,
    InstanceSegBatchPredEntity,
    InstanceSegDataEntity,
)
from otx.core.data.entity.segmentation import SegBatchDataEntity, SegBatchPredEntity, SegDataEntity
from otx.core.data.mem_cache import MemCacheHandlerSingleton
from torch import LongTensor
from torchvision import tv_tensors
from torchvision.tv_tensors import Image, Mask


@pytest.fixture(scope="session")
def fxt_data_sample() -> list[DetDataSample]:
    data_sample = DetDataSample(
        metainfo={
            "img_shape": (480, 480),
            "ori_shape": (480, 480),
            "scale_factor": (1.0, 1.0),
            "pad_shape": (480, 480),
            "ignored_labels": [],
        },
        gt_instances=InstanceData(
            bboxes=torch.Tensor([[0.0, 0.0, 240, 240], [240, 240, 480, 480]]),
            labels=torch.LongTensor([0, 1]),
        ),
    )
    return [data_sample]


@pytest.fixture(scope="session")
def fxt_det_data_entity() -> tuple[tuple, DetDataEntity, DetBatchDataEntity]:
    img_size = (64, 64)
    fake_image = torch.zeros(size=(3, *img_size), dtype=torch.uint8).numpy()
    fake_image_info = ImageInfo(img_idx=0, img_shape=img_size, ori_shape=img_size)
    fake_bboxes = tv_tensors.BoundingBoxes(data=torch.Tensor([0, 0, 5, 5]), format="xyxy", canvas_size=(10, 10))
    fake_labels = LongTensor([1])
    # define data entity
    single_data_entity = DetDataEntity(fake_image, fake_image_info, fake_bboxes, fake_labels)
    batch_data_entity = DetBatchDataEntity(
        batch_size=1,
        images=[Image(data=torch.from_numpy(fake_image))],
        imgs_info=[fake_image_info],
        bboxes=[fake_bboxes],
        labels=[fake_labels],
    )
    batch_pred_data_entity = DetBatchPredEntity(
        batch_size=1,
        images=[Image(data=torch.from_numpy(fake_image))],
        imgs_info=[fake_image_info],
        bboxes=[fake_bboxes],
        labels=[fake_labels],
        scores=[],
    )

    return single_data_entity, batch_pred_data_entity, batch_data_entity


@pytest.fixture(scope="session")
def fxt_inst_seg_data_entity() -> tuple[tuple, InstanceSegDataEntity, InstanceSegBatchDataEntity]:
    img_size = (64, 64)
    fake_image = torch.zeros(size=(3, *img_size), dtype=torch.uint8).numpy()
    fake_image_info = ImageInfo(img_idx=0, img_shape=img_size, ori_shape=img_size)
    fake_bboxes = tv_tensors.BoundingBoxes(data=torch.Tensor([0, 0, 5, 5]), format="xyxy", canvas_size=(10, 10))
    fake_labels = LongTensor([1])
    fake_masks = Mask(torch.randint(low=0, high=255, size=(1, *img_size), dtype=torch.uint8))
    fake_polygons = [Polygon(points=[1, 1, 2, 2, 3, 3, 4, 4])]
    # define data entity
    single_data_entity = InstanceSegDataEntity(
        image=fake_image,
        img_info=fake_image_info,
        bboxes=fake_bboxes,
        masks=fake_masks,
        labels=fake_labels,
        polygons=fake_polygons,
    )
    batch_data_entity = InstanceSegBatchDataEntity(
        batch_size=1,
        images=[Image(data=torch.from_numpy(fake_image))],
        imgs_info=[fake_image_info],
        bboxes=[fake_bboxes],
        labels=[fake_labels],
        masks=[fake_masks],
        polygons=[fake_polygons],
    )
    batch_pred_data_entity = InstanceSegBatchPredEntity(
        batch_size=1,
        images=[Image(data=torch.from_numpy(fake_image))],
        imgs_info=[fake_image_info],
        bboxes=[fake_bboxes],
        labels=[fake_labels],
        masks=[fake_masks],
        scores=[],
        polygons=[fake_polygons],
    )

    return single_data_entity, batch_pred_data_entity, batch_data_entity


@pytest.fixture(scope="session")
def fxt_seg_data_entity() -> tuple[tuple, SegDataEntity, SegBatchDataEntity]:
    img_size = (32, 32)
    fake_image = torch.zeros(size=(3, *img_size), dtype=torch.uint8).numpy()
    fake_image_info = ImageInfo(img_idx=0, img_shape=img_size, ori_shape=img_size)
    fake_masks = Mask(torch.randint(low=0, high=255, size=img_size, dtype=torch.uint8))
    # define data entity
    single_data_entity = SegDataEntity(fake_image, fake_image_info, fake_masks)
    batch_data_entity = SegBatchDataEntity(
        1,
        [Image(data=torch.from_numpy(fake_image))],
        [fake_image_info],
        [fake_masks],
    )
    batch_pred_data_entity = SegBatchPredEntity(
        1,
        [Image(data=torch.from_numpy(fake_image))],
        [fake_image_info],
        [],
        [fake_masks],
    )

    return single_data_entity, batch_pred_data_entity, batch_data_entity


@pytest.fixture(autouse=True)
def fxt_clean_up_mem_cache() -> None:
    """Clean up the mem-cache instance at the end of the test.

    It is required for everyone who tests model training pipeline.
    See https://github.com/openvinotoolkit/training_extensions/actions/runs/7326689283/job/19952721142?pr=2749#step:5:3098
    """
    yield
    MemCacheHandlerSingleton.delete()


# TODO(Jaeguk): Add cpu param when OTX can run integration test parallelly for each task. # noqa: TD003
@pytest.fixture(params=[pytest.param("gpu", marks=pytest.mark.gpu)])
def fxt_accelerator(request: pytest.FixtureRequest) -> str:
    return request.param
