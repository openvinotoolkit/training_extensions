# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for shared HF model-wrapper helpers."""

from __future__ import annotations

import pytest
import torch
from torchvision.ops import masks_to_boxes

from getitune.backend.huggingface.models.utils import (
    _traceable_masks_to_boxes,
    reproject_boxes_to_input_space,
    reproject_masks_to_input_space,
)
from getitune.data.entity.base import ImageInfo


def _img_info(
    ori_shape: tuple[int, int],
    scale_factor: tuple[float, float] | None,
    padding: tuple[int, int, int, int] = (0, 0, 0, 0),
) -> ImageInfo:
    return ImageInfo(  # pyrefly: ignore[no-matching-overload]
        img_idx=0,
        img_shape=ori_shape,
        ori_shape=ori_shape,
        scale_factor=scale_factor,
        padding=padding,
    )


class TestReprojectBoxesToInputSpace:
    def test_distorting_resize_scales_each_axis_independently(self) -> None:
        """Detection's base recipe: no padding, aspect ratio not preserved."""
        info = _img_info(ori_shape=(480, 720), scale_factor=(640 / 480, 640 / 720))
        boxes = torch.tensor([[100.0, 100.0, 200.0, 200.0]])

        result = reproject_boxes_to_input_space(boxes, info)

        expected_x = 100.0 * (640 / 720)
        expected_y = 100.0 * (640 / 480)
        torch.testing.assert_close(result, torch.tensor([[expected_x, expected_y, 2 * expected_x, 2 * expected_y]]))

    def test_aspect_preserving_resize_with_bottom_right_padding(self) -> None:
        """Instance segmentation's base recipe: uniform scale, pad_left=pad_top=0."""
        scale = 1024 / 720
        info = _img_info(ori_shape=(480, 720), scale_factor=(scale, scale), padding=(0, 0, 0, 341))
        boxes = torch.tensor([[0.0, 0.0, 720.0, 480.0]])

        result = reproject_boxes_to_input_space(boxes, info)

        torch.testing.assert_close(result, torch.tensor([[0.0, 0.0, 1024.0, 480.0 * scale]]))

    def test_nonzero_left_top_padding_offsets_the_box(self) -> None:
        """A centered-letterbox config would set pad_left/pad_top; the formula must honor them."""
        info = _img_info(ori_shape=(100, 100), scale_factor=(0.5, 0.5), padding=(25, 25, 25, 25))
        boxes = torch.tensor([[10.0, 10.0, 20.0, 20.0]])

        result = reproject_boxes_to_input_space(boxes, info)

        torch.testing.assert_close(result, torch.tensor([[30.0, 30.0, 35.0, 35.0]]))

    def test_empty_boxes_pass_through_unchanged(self) -> None:
        info = _img_info(ori_shape=(100, 100), scale_factor=(1.0, 1.0))
        boxes = torch.zeros((0, 4))

        result = reproject_boxes_to_input_space(boxes, info)

        assert result.shape == (0, 4)

    def test_raises_when_scale_factor_is_none(self) -> None:
        """scale_factor is None after a crop; there is nothing to reproject with."""
        info = _img_info(ori_shape=(100, 100), scale_factor=None)
        with pytest.raises(ValueError, match="scale_factor is None"):
            reproject_boxes_to_input_space(torch.tensor([[0.0, 0.0, 10.0, 10.0]]), info)


class TestReprojectMasksToInputSpace:
    def test_output_shape_matches_input_size(self) -> None:
        info = _img_info(ori_shape=(480, 720), scale_factor=(640 / 480, 640 / 720))
        masks = torch.ones((2, 480, 720), dtype=torch.uint8)

        result = reproject_masks_to_input_space(masks, info, input_size=(640, 640))

        assert result.shape == (2, 640, 640)
        assert result.dtype == torch.bool

    def test_bottom_right_padding_is_zero_outside_content(self) -> None:
        scale = 1024 / 720
        info = _img_info(ori_shape=(480, 720), scale_factor=(scale, scale), padding=(0, 0, 0, 341))
        masks = torch.ones((1, 480, 720), dtype=torch.uint8)

        result = reproject_masks_to_input_space(masks, info, input_size=(1024, 1024))

        content_h = round(480 * scale)
        assert result[0, :content_h, :].all()
        assert not result[0, content_h:, :].any()

    def test_empty_masks_return_empty_bool_tensor(self) -> None:
        info = _img_info(ori_shape=(480, 720), scale_factor=(1.0, 1.0))
        masks = torch.zeros((0, 480, 720), dtype=torch.uint8)

        result = reproject_masks_to_input_space(masks, info, input_size=(640, 640))

        assert result.shape == (0, 640, 640)
        assert result.dtype == torch.bool

    def test_raises_when_scale_factor_is_none(self) -> None:
        info = _img_info(ori_shape=(100, 100), scale_factor=None)
        with pytest.raises(ValueError, match="scale_factor is None"):
            reproject_masks_to_input_space(torch.ones((1, 100, 100), dtype=torch.uint8), info, input_size=(50, 50))


class TestTraceableMasksToBoxes:
    def test_matches_torchvision_masks_to_boxes(self) -> None:
        masks = torch.tensor(
            [
                [[0, 0, 0], [1, 1, 0], [1, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            ],
            dtype=torch.bool,
        )

        result = _traceable_masks_to_boxes(masks)
        expected = masks_to_boxes(masks)

        torch.testing.assert_close(result, expected)

    def test_empty_mask_yields_zero_box(self) -> None:
        masks = torch.zeros((1, 4, 4), dtype=torch.bool)

        result = _traceable_masks_to_boxes(masks)

        torch.testing.assert_close(result, torch.zeros((1, 4)))
