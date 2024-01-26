# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import torch

from otx.algo.hooks.recording_forward_hook import DetClassProbabilityMapHook


def test_detclassprob() -> None:
    def cls_head_forward_fn(_) -> None:
        return [torch.zeros((1, 2, 3, 3)), torch.zeros((1, 2, 6, 6))]

    num_classes = 2
    num_anchors = [1] * 10
    hook = DetClassProbabilityMapHook(
        cls_head_forward_fn,
        num_classes=num_classes,
        num_anchors=num_anchors,
    )

    assert hook.handle is None
    assert hook.records == []
    assert hook._norm_saliency_maps

    backbone_out = torch.zeros((1, 5, 2, 2))

    saliency_maps = hook.func(backbone_out)
    assert saliency_maps.size() == torch.Size([1, 2, 6, 6])

    hook.recording_forward(None, None, backbone_out)
    assert len(hook.records) == 1

    hook.reset()
    assert hook.records == []
