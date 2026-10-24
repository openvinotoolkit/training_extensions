# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for rescale_masks_to_original utility."""

import torch

from getitune.backend.openvino.models.utils import rescale_masks_to_original


class TestRescaleMasksToOriginal:
    """Tests for rescale_masks_to_original."""

    def test_empty_masks(self) -> None:
        """Empty masks return zero tensor with ori_shape."""
        masks = torch.zeros((0, 640, 640), dtype=torch.uint8)
        result = rescale_masks_to_original(masks, img_shape=(640, 640), ori_shape=(480, 640), padding=(0, 0, 0, 0))
        assert result.shape == (0, 480, 640)

    def test_same_shape_passthrough(self) -> None:
        """If img_shape == ori_shape, return masks unchanged."""
        masks = torch.ones((2, 100, 100), dtype=torch.uint8)
        result = rescale_masks_to_original(masks, img_shape=(100, 100), ori_shape=(100, 100), padding=(0, 0, 0, 0))
        assert torch.equal(result, masks)

    def test_simple_resize_no_padding(self) -> None:
        """Without padding, masks are resized directly to ori_shape."""
        # Create a 50x50 mask with a filled square in the center
        masks = torch.zeros((1, 50, 50), dtype=torch.uint8)
        masks[0, 10:40, 10:40] = 1
        result = rescale_masks_to_original(masks, img_shape=(50, 50), ori_shape=(100, 100), padding=(0, 0, 0, 0))
        assert result.shape == (1, 100, 100)
        # Center region should still be filled after upscale
        assert result[0, 25, 25] == 1
        assert result[0, 0, 0] == 0

    def test_letterbox_padding_crop(self) -> None:
        """With letterbox padding, content is cropped then resized."""
        # Simulate: original 200x100 image letterboxed to 100x100 with pad_top=25, pad_bottom=25
        # Content region is 100x100 image at rows 25:75 (50 rows height, 100 cols width)
        masks = torch.zeros((1, 100, 100), dtype=torch.uint8)
        # Place mask content in the non-padded region
        masks[0, 25:75, 0:100] = 1

        result = rescale_masks_to_original(
            masks,
            img_shape=(100, 100),
            ori_shape=(200, 100),
            padding=(0, 25, 0, 25),  # left, top, right, bottom
        )
        assert result.shape == (1, 200, 100)
        # The content (50x100) is resized to 200x100, should be all 1s
        assert result[0].sum() > 0

    def test_letterbox_asymmetric_padding(self) -> None:
        """Asymmetric padding is handled correctly."""
        # 640x640 model input, original 480x640, letterboxed with padding on bottom
        # scale = 640/640 = 1.0 for width, 640/480 = 1.333 for height -> use min = 1.0
        # Actually: fit 480x640 into 640x640: scale = min(640/480, 640/640) = 1.0
        # new_h = 480, new_w = 640, pad_top=80, pad_bottom=80
        masks = torch.zeros((1, 640, 640), dtype=torch.uint8)
        masks[0, 80:560, :] = 1  # content region

        result = rescale_masks_to_original(
            masks,
            img_shape=(640, 640),
            ori_shape=(480, 640),
            padding=(0, 80, 0, 80),
        )
        assert result.shape == (1, 480, 640)
        # Content was 480x640, resized to 480x640 — should be all 1s
        assert result[0].sum() == 480 * 640
