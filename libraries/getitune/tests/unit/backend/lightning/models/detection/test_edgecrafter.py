# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Test of EdgeCrafter."""

import torch
from torchvision import tv_tensors

from getitune.backend.lightning.models.base import DataInputParams
from getitune.backend.lightning.models.detection.edgecrafter import EdgeCrafter
from getitune.data.entity.sample import SampleBatch
from getitune.types import LabelInfo


class TestEdgeCrafter:
    def test_customize_inputs_handles_degenerate_empty_bboxes(self, mocker):
        """A (1, 0)-shaped empty bboxes tensor must not reach box_convert and crash."""
        label_info = LabelInfo(["a", "b", "c"], ["0", "1", "2"], [["a", "b", "c"]])
        mocker.patch(
            "getitune.backend.lightning.models.detection.edgecrafter.EdgeCrafter._create_model",
            return_value=mocker.MagicMock(),
        )
        model = EdgeCrafter(
            model_name="edgecrafter_s",
            label_info=label_info,
            data_input_params=DataInputParams((320, 320), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        )
        valid_bboxes = tv_tensors.BoundingBoxes(  # pyrefly: ignore[no-matching-overload]
            torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
            format="XYXY",
            canvas_size=(320, 320),
        )
        # validate_bboxes only inspects the first non-None entry, so the degenerate
        # shape must be on a later sample to reach the model's defensive guard.
        degenerate_bboxes = tv_tensors.BoundingBoxes(  # pyrefly: ignore[no-matching-overload]
            torch.zeros((1, 0), dtype=torch.float32),
            format="XYXY",
            canvas_size=(320, 320),
        )
        entity = SampleBatch(
            images=torch.randn(2, 3, 320, 320),
            bboxes=[valid_bboxes, degenerate_bboxes],
            labels=[torch.zeros(1, dtype=torch.long), torch.zeros(0, dtype=torch.long)],
        )

        result = model._customize_inputs(entity)

        assert result["targets"][1]["boxes"].numel() == 0
