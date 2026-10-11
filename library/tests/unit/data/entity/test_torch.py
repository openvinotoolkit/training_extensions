# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Unit tests of sample batch data entity."""

from unittest.mock import Mock

import torch
from datumaro.experimental.fields import ImageInfo as DmImageInfo
from torch import LongTensor
from torchvision import tv_tensors

from getitune.data.dataset.base import _default_collate_fn
from getitune.data.entity.base import ImageInfo
from getitune.data.entity.sample import (
    BaseSample,
    DetectionSample,
    InstanceSegmentationSample,
    SampleBatch,
)


class TestDetectionSampleNoneAnnotations:
    """Test DetectionSample.__post_init__ converts None fields to empty tensors."""

    def test_detection_sample_none_label(self) -> None:
        """Verify label=None is converted to an empty uint8 tensor and collation works."""
        img_size = (64, 64)
        sample = DetectionSample(
            image=tv_tensors.Image(torch.randint(0, 256, (3, *img_size), dtype=torch.uint8)),
            dm_image_info=DmImageInfo(height=img_size[0], width=img_size[1]),
            bboxes=torch.zeros((0, 4), dtype=torch.float32),
            label=None,
        )
        # label should be converted to an empty tensor with uint8 dtype
        assert isinstance(sample.label, torch.Tensor)
        assert sample.label.shape == (0,)
        assert sample.label.dtype == torch.uint8

        # bboxes should be converted to BoundingBoxes
        assert isinstance(sample.bboxes, tv_tensors.BoundingBoxes)
        assert sample.bboxes.shape == (0, 4)

    def test_detection_sample_none_label_collation(self) -> None:
        """Verify collation works when mixing annotated and unannotated detection samples."""
        img_size = (32, 32)
        annotated = DetectionSample(
            image=tv_tensors.Image(torch.rand(3, *img_size, dtype=torch.float32)),
            dm_image_info=DmImageInfo(height=img_size[0], width=img_size[1]),
            bboxes=tv_tensors.BoundingBoxes(  # type: ignore[call-overload]
                torch.tensor([[5, 5, 20, 20]], dtype=torch.float32),
                format=tv_tensors.BoundingBoxFormat.XYXY,
                canvas_size=img_size,
            ),
            label=torch.tensor([1], dtype=torch.uint8),
        )
        unannotated = DetectionSample(
            image=tv_tensors.Image(torch.rand(3, *img_size, dtype=torch.float32)),
            dm_image_info=DmImageInfo(height=img_size[0], width=img_size[1]),
            bboxes=torch.zeros((0, 4), dtype=torch.float32),
            label=None,
        )
        batch = _default_collate_fn([annotated, unannotated])
        assert isinstance(batch, SampleBatch)
        assert batch.batch_size == 2
        # Labels should be cast to long by the collate function
        assert batch.labels is not None
        assert all(label.dtype == torch.long for label in batch.labels)

    def test_detection_sample_degenerate_empty_bboxes_reshaped(self) -> None:
        """Verify a degenerate (1, 0) empty bboxes tensor is reshaped to (0, 4)."""
        img_size = (64, 64)
        sample = DetectionSample(
            image=tv_tensors.Image(torch.randint(0, 256, (3, *img_size), dtype=torch.uint8)),
            dm_image_info=DmImageInfo(height=img_size[0], width=img_size[1]),
            bboxes=torch.zeros((1, 0), dtype=torch.float32),
            label=None,
        )
        assert isinstance(sample.bboxes, tv_tensors.BoundingBoxes)
        assert sample.bboxes.shape == (0, 4)


class TestInstanceSegmentationSampleNoneAnnotations:
    """Test InstanceSegmentationSample.__post_init__ converts None fields to empty tensors."""

    def test_instance_seg_sample_none_fields(self) -> None:
        """Verify label=None and masks=None are converted to correctly shaped empty tensors."""
        img_size = (48, 64)
        sample = InstanceSegmentationSample(
            image=tv_tensors.Image(torch.randint(0, 256, (3, *img_size), dtype=torch.uint8)),
            dm_image_info=DmImageInfo(height=img_size[0], width=img_size[1]),
            bboxes=torch.zeros((0, 4), dtype=torch.float32),
            label=None,
            masks=None,
        )
        # label should be an empty uint8 tensor
        assert isinstance(sample.label, torch.Tensor)
        assert sample.label.shape == (0,)
        assert sample.label.dtype == torch.uint8

        # masks should be a Mask with shape (0, H, W) matching the image spatial dims
        assert isinstance(sample.masks, tv_tensors.Mask)
        assert sample.masks.shape == (0, *img_size)
        assert sample.masks.dtype == torch.uint8

        # bboxes should be converted to BoundingBoxes
        assert isinstance(sample.bboxes, tv_tensors.BoundingBoxes)
        assert sample.bboxes.shape == (0, 4)

    def test_instance_seg_sample_none_fields_collation(self) -> None:
        """Verify collation works when mixing annotated and unannotated IS samples."""
        img_size = (32, 32)
        annotated = InstanceSegmentationSample(
            image=tv_tensors.Image(torch.rand(3, *img_size, dtype=torch.float32)),
            dm_image_info=DmImageInfo(height=img_size[0], width=img_size[1]),
            bboxes=tv_tensors.BoundingBoxes(  # type: ignore[call-overload]
                torch.tensor([[5, 5, 20, 20]], dtype=torch.float32),
                format=tv_tensors.BoundingBoxFormat.XYXY,
                canvas_size=img_size,
            ),
            label=torch.tensor([1], dtype=torch.uint8),
            masks=tv_tensors.Mask(torch.ones((1, *img_size), dtype=torch.uint8)),
        )
        unannotated = InstanceSegmentationSample(
            image=tv_tensors.Image(torch.rand(3, *img_size, dtype=torch.float32)),
            dm_image_info=DmImageInfo(height=img_size[0], width=img_size[1]),
            bboxes=torch.zeros((0, 4), dtype=torch.float32),
            label=None,
            masks=None,
        )
        batch = _default_collate_fn([annotated, unannotated])
        assert isinstance(batch, SampleBatch)
        assert batch.batch_size == 2
        # Labels should be cast to long by the collate function
        assert batch.labels is not None
        assert all(label.dtype == torch.long for label in batch.labels)
        # Masks should be present for both samples
        assert batch.masks is not None
        assert len(batch.masks) == 2

    def test_instance_seg_sample_degenerate_empty_bboxes_reshaped(self) -> None:
        """Verify a degenerate (1, 0) empty bboxes tensor is reshaped to (0, 4)."""
        img_size = (48, 64)
        sample = InstanceSegmentationSample(
            image=tv_tensors.Image(torch.randint(0, 256, (3, *img_size), dtype=torch.uint8)),
            dm_image_info=DmImageInfo(height=img_size[0], width=img_size[1]),
            bboxes=torch.zeros((1, 0), dtype=torch.float32),
            label=None,
            masks=None,
        )
        assert isinstance(sample.bboxes, tv_tensors.BoundingBoxes)
        assert sample.bboxes.shape == (0, 4)


class TestCollateFn:
    def test_collate_fn(self) -> None:
        """Test _default_collate_fn function."""
        # Create mock samples with required attributes
        samples = []
        for i in range(3):
            sample = Mock(spec=BaseSample)
            # Use float32 images since _default_collate_fn expects tensors
            sample.image = tv_tensors.Image(torch.randn(3, 224, 224))
            sample.img_info = ImageInfo(img_idx=i, img_shape=(224, 224), ori_shape=(224, 224))
            sample.bboxes = tv_tensors.BoundingBoxes(
                data=torch.Tensor([0, 0, 50, 50]),
                format="xywh",
                canvas_size=(224, 224),
            )
            sample.label = LongTensor([1])
            sample.masks = None
            sample.keypoints = None
            samples.append(sample)

        data_batch = _default_collate_fn(samples)
        assert len(data_batch.imgs_info) == len(data_batch.images)
        assert isinstance(data_batch, SampleBatch)
        for field in SampleBatch.__dataclass_fields__:
            assert hasattr(data_batch, field), f"Field {field} is missing in the collated batch"
