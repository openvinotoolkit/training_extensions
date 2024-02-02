# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Utils used for XAI."""

from __future__ import annotations

from otx.algo.hooks.recording_forward_hook import BaseRecordingForwardHook, MaskRCNNRecordingForwardHook
from otx.core.config.explain import ExplainConfig


def get_processed_saliency_maps(
    raw_saliency_maps: list,
    explain_config: ExplainConfig,
    predictions: list | None,
    explain_hook: BaseRecordingForwardHook,
) -> list:
    """Implement saliency map filtering and post-processing."""
    if predictions is not None and isinstance(explain_hook, MaskRCNNRecordingForwardHook):
        # TODO: It is a temporary workaround. This function will be removed after we # noqa: TD003, TD002
        # refactor XAI logics into `OTXModel.forward_explain()`.

        # Mask-RCNN case, receive saliency maps from predictions
        raw_saliency_maps = explain_hook.func(predictions)
    return raw_saliency_maps
