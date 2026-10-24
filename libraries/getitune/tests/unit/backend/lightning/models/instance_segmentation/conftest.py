# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Test of custom algo modules of getitune Detection task."""

import pytest
import torch
from torchvision import tv_tensors

from getitune.data.entity.base import ImageInfo
from getitune.data.entity.sample import SampleBatch


@pytest.fixture
def fxt_instance_seg_batch(batch_size: int = 2) -> SampleBatch:
    """Create a mock instance segmentation batch for testing."""
    images = [torch.randn(3, 640, 640), torch.randn(3, 640, 640)]

    # Create bounding boxes and labels for each image
    bboxes = [
        tv_tensors.BoundingBoxes(
            torch.tensor([[100, 100, 300, 300], [200, 200, 400, 400]], dtype=torch.float32),
            format=tv_tensors.BoundingBoxFormat.XYXY,
            canvas_size=(640, 640),
        ),
        tv_tensors.BoundingBoxes(
            torch.tensor([[150, 150, 350, 350]], dtype=torch.float32),
            format=tv_tensors.BoundingBoxFormat.XYXY,
            canvas_size=(640, 640),
        ),
    ]

    labels = [
        torch.tensor([0, 1], dtype=torch.long),
        torch.tensor([2], dtype=torch.long),
    ]

    # Create masks for each instance
    masks = [
        tv_tensors.Mask(torch.zeros((2, 640, 640), dtype=torch.uint8)),
        tv_tensors.Mask(torch.zeros((1, 640, 640), dtype=torch.uint8)),
    ]

    imgs_info = [
        ImageInfo(img_idx=0, img_shape=(640, 640), ori_shape=(640, 640)),
        ImageInfo(img_idx=1, img_shape=(640, 640), ori_shape=(640, 640)),
    ]

    return SampleBatch(
        images=torch.stack(images),
        bboxes=bboxes,
        labels=labels,
        masks=masks,
        imgs_info=imgs_info,
    )
