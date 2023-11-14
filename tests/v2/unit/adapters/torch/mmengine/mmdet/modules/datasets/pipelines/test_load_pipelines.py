"""Unit tests of otx/v2/adapters/torch/mmengine/mmdet/modules/datasets/pipeline/load_pipelines.py."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from mmdet.utils import register_all_modules

from otx.v2.adapters.torch.mmengine.mmdet.modules.datasets.pipelines import (
    LoadResizeDataFromOTXDataset,
    LoadAnnotationFromOTXDataset
)
from otx.v2.api.entities.task_type import TaskType
from otx.v2.adapters.datumaro.caching import MemCacheHandlerSingleton
from tests.v2.unit.adapters.torch.mmengine.mmdet.test_helpers import generate_det_dataset


def test_load_resize_data_from_otx_dataset_call(mocker):
    """Test LoadResizeDataFromOTXDataset."""
    # Temporary solution for registry confusion.
    register_all_modules(init_default_scope=True)
    otx_dataset, labels = generate_det_dataset(
        TaskType.DETECTION,  # covers det & iseg format both
    )
    for item in otx_dataset:
        sample_item = item
        break
    width, height, channels = sample_item.media.data.shape
    MemCacheHandlerSingleton.create("singleprocessing", width * height * channels)
    operation = LoadResizeDataFromOTXDataset(
        load_ann_cfg=dict(
            type="LoadAnnotationFromOTXDataset",
            domain="detection",
            with_bbox=True,
            with_mask=False,
            poly2mask=False,
        ),
        resize_cfg=dict(type="Resize", scale=(32, 16), keep_ratio=False),  # 320x320 -> 16x32
    )
    src_dict = dict(
        dataset_item=sample_item,
        width=width,
        height=height,
        index=0,
        ann_info=dict(label_list=labels),
        bbox_fields=[],
        mask_fields=[],
    )
    dst_dict = operation(src_dict)
    assert dst_dict["ori_shape"][0] == 720
    assert dst_dict["img_shape"][0] == 16  # height
    assert dst_dict["img"].shape[:2] == dst_dict["img_shape"]
    operation._load_img = mocker.MagicMock()
    dst_dict_from_cache = operation(src_dict)
    assert operation._load_img.call_count == 0  # _load_img() should not be called
    assert np.array_equal(dst_dict["img"], dst_dict_from_cache["img"])
    assert (dst_dict["gt_bboxes_labels"] == dst_dict_from_cache["gt_bboxes_labels"]).all()
    assert (dst_dict["gt_bboxes"] == dst_dict_from_cache["gt_bboxes"]).all()


def test_load_resize_data_from_otx_dataset_downscale_only(mocker):
    """Test LoadResizeDataFromOTXDataset."""
    otx_dataset, labels = generate_det_dataset(
        TaskType.DETECTION,  # covers det & iseg format both
    )
    for item in otx_dataset:
        sample_item = item
        break
    width, height, channels = sample_item.media.data.shape
    MemCacheHandlerSingleton.create("singleprocessing", width * height * channels)
    operation = LoadResizeDataFromOTXDataset(
        load_ann_cfg=dict(
            type="LoadAnnotationFromOTXDataset",
            domain="instance_segmentation",
            with_bbox=True,
            with_mask=False,
            poly2mask=False,
        ),
        resize_cfg=dict(type="Resize", scale=(640, 640), downscale_only=True),  # 320x320 -> 16x32
    )
    src_dict = dict(
        dataset_item=sample_item,
        width=width,
        height=height,
        index=0,
        ann_info=dict(label_list=labels),
        bbox_fields=[],
        mask_fields=[],
    )
    dst_dict = operation(src_dict)
    assert dst_dict["ori_shape"][0] == 720
    assert dst_dict["img_shape"][0] == 640  # Skipped upscale
    assert dst_dict["img"].shape[:2] == dst_dict["img_shape"]
    operation._load_img_op = mocker.MagicMock()
    dst_dict_from_cache = operation(src_dict)
    assert operation._load_img_op.call_count == 0  # _load_img() should not be called
    assert np.array_equal(dst_dict["img"], dst_dict_from_cache["img"])
    assert (dst_dict["gt_bboxes_labels"] == dst_dict_from_cache["gt_bboxes_labels"]).all()
    assert (dst_dict["gt_bboxes"] == dst_dict_from_cache["gt_bboxes"]).all()


def test_load_annotation_from_otx_dataset():
    """Test LoadAnnotationFromOTXDataset."""
    operation = LoadAnnotationFromOTXDataset(
        with_bbox=True,
        with_label=True,
        with_mask=False,
        domain="instance_segmentation",
    )
    otx_dataset, labels = generate_det_dataset(
        TaskType.DETECTION,
    )
    for item in otx_dataset:
        sample_item = item
        break
    width, height, channels = sample_item.media.data.shape

    src_dict = dict(
        dataset_item=sample_item,
        WIdth=width,
        height=height,
        index=0,
        ann_info=dict(label_list=labels),
        bbox_fields=[],
        mask_fields=[],
    )
    dst_dict = operation(src_dict)
    assert dst_dict["bbox_fields"] == ["gt_bboxes"]
    assert "gt_bboxes" in dst_dict
    assert "gt_ann_ids" in dst_dict
    assert "gt_ignore_flags" in dst_dict
    assert "gt_bboxes_labels" in dst_dict
