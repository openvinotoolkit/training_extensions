# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Utils used for XAI."""

from __future__ import annotations

from otx.algo.hooks.recording_forward_hook import MaskRCNNRecordingForwardHook
from otx.core.config.explain import ExplainConfig
from otx.core.data.entity.instance_segmentation import InstanceSegBatchPredEntity


def get_processed_saliency_maps(
    raw_saliency_maps: list,
    explain_config: ExplainConfig,
    predictions: list | None,
) -> list:
    """Implement saliency map filtering and post-processing."""
    if predictions is not None and isinstance(predictions[0], InstanceSegBatchPredEntity):
        # Mask-RCNN case, receive saliency maps from predictions
        raw_saliency_maps = MaskRCNNRecordingForwardHook.get_sal_map_from_preds(
            predictions,
            explain_config.num_classes,
        )
    return raw_saliency_maps
