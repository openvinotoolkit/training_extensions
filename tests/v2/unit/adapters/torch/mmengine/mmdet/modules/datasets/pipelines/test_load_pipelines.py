# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from otx.v2.adapters.torch.mmengine.mmdet.modules.datasets.pipelines import (
    LoadResizeDataFromOTXDataset,
    LoadAnnotationFromOTXDataset
)
from otx.v2.api.entities.task_type import TaskType
from otx.v2.adapters.datumaro.caching import MemCacheHandlerSingleton
from tests.v2.unit.adapters.torch.mmengine.mmdet.test_helpers import generate_det_dataset


def test_load_resize_data_from_otx_dataset_call(mocker):
    """Test LoadResizeDataFromOTXDataset."""
    otx_dataset, labels = generate_det_dataset(
        TaskType.INSTANCE_SEGMENTATION,  # covers det & iseg format both
        image_width=320,
        image_height=320,
    )
    MemCacheHandlerSingleton.create("singleprocessing", otx_dataset[0].numpy.size)
    op = LoadResizeDataFromOTXDataset(
        load_ann_cfg=dict(
            type="LoadAnnotationFromOTXDataset",
            domain="instance_segmentation",
            with_bbox=True,
            with_mask=True,
            poly2mask=False,
        ),
        resize_cfg=dict(type="Resize", scale=(32, 16), keep_ratio=False),  # 320x320 -> 16x32
    )
    src_dict = dict(
        dataset_item=otx_dataset[0],
        width=otx_dataset[0].width,
        height=otx_dataset[0].height,
        index=0,
        ann_info=dict(label_list=labels),
        bbox_fields=[],
        mask_fields=[],
    )
    dst_dict = op(src_dict)
    assert dst_dict["ori_shape"][0] == 320
    assert dst_dict["img_shape"][0] == 16  # height
    assert dst_dict["img"].shape[:2] == dst_dict["img_shape"]
    assert dst_dict["gt_masks"].width == 32
    assert dst_dict["gt_masks"].height == 16
    op._load_img = mocker.MagicMock()
    dst_dict_from_cache = op(src_dict)
    assert op._load_img.call_count == 0  # _load_img() should not be called
    assert np.array_equal(dst_dict["img"], dst_dict_from_cache["img"])
    assert (dst_dict["gt_bboxes_labels"] == dst_dict_from_cache["gt_bboxes_labels"]).all()
    assert (dst_dict["gt_bboxes"] == dst_dict_from_cache["gt_bboxes"]).all()
    assert dst_dict["gt_masks"] == dst_dict_from_cache["gt_masks"]


def test_load_resize_data_from_otx_dataset_downscale_only(mocker):
    """Test LoadResizeDataFromOTXDataset."""
    otx_dataset, labels = generate_det_dataset(
        TaskType.INSTANCE_SEGMENTATION,  # covers det & iseg format both
        image_width=320,
        image_height=320,
    )
    MemCacheHandlerSingleton.create("singleprocessing", otx_dataset[0].numpy.size)
    op = LoadResizeDataFromOTXDataset(
        load_ann_cfg=dict(
            type="LoadAnnotationFromOTXDataset",
            domain="instance_segmentation",
            with_bbox=True,
            with_mask=True,
            poly2mask=False,
        ),
        resize_cfg=dict(type="Resize", scale=(640, 640), downscale_only=True),  # 320x320 -> 16x32
    )
    src_dict = dict(
        dataset_item=otx_dataset[0],
        width=otx_dataset[0].width,
        height=otx_dataset[0].height,
        index=0,
        ann_info=dict(label_list=labels),
        bbox_fields=[],
        mask_fields=[],
    )
    dst_dict = op(src_dict)
    assert dst_dict["ori_shape"][0] == 320
    assert dst_dict["img_shape"][0] == 320  # Skipped upscale
    assert dst_dict["img"].shape == dst_dict["img_shape"]
    op._load_img_op = mocker.MagicMock()
    dst_dict_from_cache = op(src_dict)
    assert op._load_img_op.call_count == 0  # _load_img() should not be called
    assert np.array_equal(dst_dict["img"], dst_dict_from_cache["img"])
    assert (dst_dict["gt_bboxes_labels"] == dst_dict_from_cache["gt_bboxes_labels"]).all()
    assert (dst_dict["gt_bboxes"] == dst_dict_from_cache["gt_bboxes"]).all()
    assert dst_dict["gt_masks"] == dst_dict_from_cache["gt_masks"]


def test_load_annotation_from_otx_dataset():
    """Test LoadAnnotationFromOTXDataset."""
    op = LoadAnnotationFromOTXDataset(
        with_bbox=True,
        with_label=True,
        with_mask=True,
        domain="instance_segmentation",
    )
    otx_dataset, labels = generate_det_dataset(
        TaskType.INSTANCE_SEGMENTATION,
        image_width=320,
        image_height=320,
    )

    src_dict = dict(
        dataset_item=otx_dataset[0],
        width=otx_dataset[0].width,
        height=otx_dataset[0].height,
        index=0,
        ann_info=dict(label_list=labels),
        bbox_fields=[],
        mask_fields=[],
    )
    dst_dict = op(src_dict)
    assert dst_dict["bbox_fields"] == ["gt_bboxes"]
    assert "gt_bboxes" in dst_dict
    assert "gt_ann_ids" in dst_dict
    assert "gt_ignore_flags" in dst_dict
    assert "gt_bboxes_labels" in dst_dict
    assert dst_dict["mask_fields"] == ["gt_masks"]
    assert "gt_masks" in dst_dict
