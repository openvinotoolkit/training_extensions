# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import torch
from otx.algo.hooks.recording_forward_hook import ReciproCAMHook


def test_reciprocam():
    def cls_head_forward_fn(_):
        return torch.zeros((25, 2))

    num_classes = 2
    optimize_gap = False
    hook = ReciproCAMHook(
        cls_head_forward_fn,
        num_classes=num_classes,
        optimize_gap=optimize_gap,
    )

    assert hook.handle is None
    assert hook.records == []
    assert hook._norm_saliency_maps

    feature_map = torch.zeros((1, 10, 5, 5))

    saliency_maps = hook.func(feature_map)
    assert saliency_maps.size() == torch.Size([1, 2, 5, 5])

    hook.recording_forward(None, None, feature_map)
    assert len(hook.records) == 1

    hook.reset()
    assert hook.records == []
