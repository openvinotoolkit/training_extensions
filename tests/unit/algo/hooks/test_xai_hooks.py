# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import torch
from otx.algo.hooks.recording_forward_hook import (
    ViTReciproCAMHook,
)


def test_vitreciprocam() -> None:
    def cls_head_forward_fn(_) -> None:
        return torch.zeros((196, 2))

    num_classes = 2
    hook = ViTReciproCAMHook(
        cls_head_forward_fn,
        num_classes=num_classes,
    )

    assert hook.handle is None
    assert hook.records == []
    assert hook._norm_saliency_maps

    feature_map = torch.zeros((1, 197, 192))

    saliency_maps = hook.func(feature_map)
    assert saliency_maps.size() == torch.Size([1, 2, 14, 14])

    hook.recording_forward(None, None, feature_map)
    assert len(hook.records) == 1

    hook.reset()
    assert hook.records == []
