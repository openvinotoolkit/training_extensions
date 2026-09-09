# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Timm-specific optimizer selection and preprocessing helpers."""

from __future__ import annotations

from typing import cast

import timm

from getitune.backend.lightning.models.base import DataInputParams

# timm architectures whose ops (e.g. non-contiguous transpose/reshape in attention blocks,
# adaptive pooling with non-divisible output sizes, or ops that lower to sequence
# constructs like `unbind`) the legacy TorchScript-based ONNX exporter cannot reliably
# trace or decompose. These require the newer Dynamo-based `torch.onnx.export` path instead
# of the legacy exporter used by default for other timm backbones.
DYNAMO_REQUIRED_ARCHITECTURES = ("naflexvit", "nfnet", "volo", "halo", "csatv2")


def get_preprocessing_params(backbone_name: str) -> DataInputParams | dict[str, DataInputParams]:
    """Get default preprocessing parameters for the backbone."""
    cfg = timm.get_pretrained_cfg(backbone_name)
    if cfg is None:
        msg = f"Backbone {backbone_name} does not have a default preprocessing configuration."
        raise ValueError(msg)
    return DataInputParams(
        input_size=cfg.input_size[-2:],
        mean=cast("tuple[float, float, float]", cfg.mean),
        std=cast("tuple[float, float, float]", cfg.std),
    )
