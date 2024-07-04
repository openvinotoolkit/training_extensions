# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Test Max Iou Assigner ."""

import torch
from otx.algo.common.utils.assigners.max_iou_assigner import perm_repeat_bboxes


def test_perm_repeat_bboxes() -> None:
    sample = torch.randn(1, 4)
    inputs = torch.stack([sample for i in range(10)])
    assert perm_repeat_bboxes(inputs, {}).shape == torch.Size([10, 1, 4])
