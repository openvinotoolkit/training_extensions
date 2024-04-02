# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Unit tests of detection data transform."""

import torch
import ast
from torchvision import tv_tensors

from otx.core.data.transform.detection import MinIoURandomCrop
from torchvision.ops.boxes import box_iou
import pytest


class TestMinIoURandomCrop:
    @pytest.mark.parametrize("options", [[0.5, 0.9], [2.0]])
    def test_get_params(self, options: list[float]) -> None:
        """Test _get_params."""
        orig_h, orig_w = size = (24, 32)
        bboxes = tv_tensors.BoundingBoxes(
            torch.tensor([[1, 1, 10, 10], [20, 20, 23, 23], [1, 20, 10, 23], [20, 1, 23, 10]]),
            format="XYXY",
            canvas_size=size,
        )
        sample = [bboxes]

        transform = MinIoURandomCrop(sampler_options=options)

        n_samples = 5
        for _ in range(n_samples):

            params = transform._get_params(sample)

            if options == [2.0]:
                assert len(params) == 0
                return

            assert len(params["is_within_crop_area"]) > 0
            assert params["is_within_crop_area"].dtype == torch.bool

            assert int(transform.min_scale * orig_h) <= params["height"] <= int(transform.max_scale * orig_h)
            assert int(transform.min_scale * orig_w) <= params["width"] <= int(transform.max_scale * orig_w)

            left, top = params["left"], params["top"]
            new_h, new_w = params["height"], params["width"]
            ious = box_iou(
                bboxes,
                torch.tensor([[left, top, left + new_w, top + new_h]], dtype=bboxes.dtype, device=bboxes.device),
            )
            assert ious.max() >= options[0] or ious.max() >= options[1], f"{ious} vs {options}"
