import numpy as np
import pytest
from PIL import Image
from typing import Iterator, List, Optional, Sequence, Tuple

from otx.algorithms.detection.adapters.mmdet.datasets.pipelines import (
    LoadResizeDataFromOTXDataset,
    ResizeTo,
)
from otx.api.entities.model_template import TaskType
from otx.core.data.caching import MemCacheHandlerSingleton
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.algorithms.detection.test_helpers import generate_det_dataset


@e2e_pytest_unit
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
        resize_cfg=dict(type="ResizeTo", img_scale=(32, 16), keep_ratio=False),  # 320x320 -> 16x32
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
    assert dst_dict["img"].shape == dst_dict["img_shape"]
    assert dst_dict["gt_masks"].width == 32
    assert dst_dict["gt_masks"].height == 16
    op._load_img = mocker.MagicMock()
    dst_dict_from_cache = op(src_dict)
    assert op._load_img.call_count == 0  # _load_img() should not be called
    assert np.array_equal(dst_dict["img"], dst_dict_from_cache["img"])
    assert (dst_dict["gt_labels"] == dst_dict_from_cache["gt_labels"]).all()
    assert (dst_dict["gt_bboxes"] == dst_dict_from_cache["gt_bboxes"]).all()
    assert dst_dict["gt_masks"] == dst_dict_from_cache["gt_masks"]


@e2e_pytest_unit
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
        resize_cfg=dict(type="ResizeTo", img_scale=(640, 640), downscale_only=True),  # 320x320 -> 16x32
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
    assert (dst_dict["gt_labels"] == dst_dict_from_cache["gt_labels"]).all()
    assert (dst_dict["gt_bboxes"] == dst_dict_from_cache["gt_bboxes"]).all()
    assert dst_dict["gt_masks"] == dst_dict_from_cache["gt_masks"]


@e2e_pytest_unit
def test_resize_to(mocker):
    """Test ResizeTo."""
    src_dict = dict(
        img=np.random.randint(0, 10, (16, 16, 3), dtype=np.uint8),
        img_fields=["img"],
        ori_shape=(16, 16),
        img_shape=(16, 16),
    )
    # Test downscale
    op = ResizeTo(img_scale=(4, 4))
    dst_dict = op(src_dict)
    assert dst_dict["ori_shape"][0] == 16
    assert dst_dict["img_shape"][0] == 4
    assert dst_dict["img"].shape == dst_dict["img_shape"]
    # Test upscale from output
    op = ResizeTo(img_scale=(8, 8))
    dst_dict = op(dst_dict)
    assert dst_dict["ori_shape"][0] == 16
    assert dst_dict["img_shape"][0] == 8
    assert dst_dict["img"].shape == dst_dict["img_shape"]
    # Test same size from output
    op = ResizeTo(img_scale=(8, 8))
    op._resize_img = mocker.MagicMock()
    dst_dict = op(dst_dict)
    assert dst_dict["ori_shape"][0] == 16
    assert dst_dict["img_shape"][0] == 8
    assert op._resize_img.call_count == 0  # _resize_img() should not be called
