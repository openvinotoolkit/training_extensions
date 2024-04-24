# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""LiteHRNet model implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from otx.algo.segmentation.backbones import LiteHRNet
from otx.algo.segmentation.heads import CustomFCNHead
from otx.algo.utils.support_otx_v1 import OTXv1Helper
from otx.core.model.segmentation import TorchVisionCompatibleModel

from .base_model import BaseSegmNNModel

if TYPE_CHECKING:
    from torch import nn


class LiteHRNetS(BaseSegmNNModel):
    """LiteHRNetS Model."""

    default_backbone_configuration: ClassVar[dict[str, Any]] = {
        "norm_cfg": {"type": "BN", "requires_grad": True},
        "norm_eval": False,
        "extra": {
            "stem": {
                "stem_channels": 32,
                "out_channels": 32,
                "expand_ratio": 1,
                "strides": [2, 2],
                "extra_stride": True,
                "input_norm": False,
            },
            "num_stages": 2,
            "stages_spec": {
                "num_modules": [4, 4],
                "num_branches": [2, 3],
                "num_blocks": [2, 2],
                "module_type": ["LITE", "LITE"],
                "with_fuse": [True, True],
                "reduce_ratios": [8, 8],
                "num_channels": [[60, 120], [60, 120, 240]],
            },
            "out_modules": {
                "conv": {"enable": False, "channels": 160},
                "position_att": {"enable": False, "key_channels": 64, "value_channels": 240, "psp_size": [1, 3, 6, 8]},
                "local_att": {"enable": False},
            },
            "out_aggregator": {"enable": False},
            "add_input": False,
        },
        "pretrained_weights": "https://storage.openvinotoolkit.org/repositories/openvino_training_extensions/models/custom_semantic_segmentation/litehrnetsv2_imagenet1k_rsc.pth",
    }
    default_decode_head_configuration: ClassVar[dict[str, Any]] = {
        "norm_cfg": {"type": "BN", "requires_grad": True},
        "in_channels": [60, 120, 240],
        "in_index": [0, 1, 2],
        "input_transform": "multiple_select",
        "channels": 60,
        "kernel_size": 1,
        "num_convs": 1,
        "concat_input": False,
        "enable_aggregator": True,
        "aggregator_merge_norm": "None",
        "aggregator_use_concat": False,
        "dropout_ratio": -1,
        "align_corners": False,
    }

    @property
    def ignore_scope(self) -> dict[str, str | dict[str, list[str]]]:
        """The ignored scope for LiteHRNetS."""
        ignored_scope_names = [
            "/model/backbone/stage0/stage0.0/layers/layers.0/cross_resolution_weighting/Mul",
            "/model/backbone/stage0/stage0.0/layers/layers.0/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage0/stage0.0/layers/layers.1/cross_resolution_weighting/Mul",
            "/model/backbone/stage0/stage0.0/layers/layers.1/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage0/stage0.0/Add_1",
            "/model/backbone/stage0/stage0.1/layers/layers.0/cross_resolution_weighting/Mul",
            "/model/backbone/stage0/stage0.1/layers/layers.0/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage0/stage0.1/layers/layers.1/cross_resolution_weighting/Mul",
            "/model/backbone/stage0/stage0.1/layers/layers.1/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage0/stage0.1/Add_1",
            "/model/backbone/stage0/stage0.2/layers/layers.0/cross_resolution_weighting/Mul",
            "/model/backbone/stage0/stage0.2/layers/layers.0/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage0/stage0.2/layers/layers.1/cross_resolution_weighting/Mul",
            "/model/backbone/stage0/stage0.2/layers/layers.1/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage0/stage0.2/Add_1",
            "/model/backbone/stage0/stage0.3/layers/layers.0/cross_resolution_weighting/Mul",
            "/model/backbone/stage0/stage0.3/layers/layers.0/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage0/stage0.3/layers/layers.1/cross_resolution_weighting/Mul",
            "/model/backbone/stage0/stage0.3/layers/layers.1/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage0/stage0.3/Add_1",
            "/model/backbone/stage1/stage1.0/layers/layers.0/cross_resolution_weighting/Mul",
            "/model/backbone/stage1/stage1.0/layers/layers.0/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage1/stage1.0/layers/layers.0/cross_resolution_weighting/Mul_2",
            "/model/backbone/stage1/stage1.0/layers/layers.1/cross_resolution_weighting/Mul",
            "/model/backbone/stage1/stage1.0/layers/layers.1/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage1/stage1.0/Add_1",
            "/model/backbone/stage1/stage1.0/layers/layers.1/cross_resolution_weighting/Mul_2",
            "/model/backbone/stage1/stage1.0/Add_2",
            "/model/backbone/stage1/stage1.0/Add_5",
            "/model/backbone/stage1/stage1.1/layers/layers.0/cross_resolution_weighting/Mul",
            "/model/backbone/stage1/stage1.1/layers/layers.0/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage1/stage1.1/layers/layers.0/cross_resolution_weighting/Mul_2",
            "/model/backbone/stage1/stage1.1/layers/layers.1/cross_resolution_weighting/Mul",
            "/model/backbone/stage1/stage1.1/layers/layers.1/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage1/stage1.1/Add_1",
            "/model/backbone/stage1/stage1.1/layers/layers.1/cross_resolution_weighting/Mul_2",
            "/model/backbone/stage1/stage1.1/Add_2",
            "/model/backbone/stage1/stage1.1/Add_5",
            "/model/backbone/stage1/stage1.2/layers/layers.0/cross_resolution_weighting/Mul",
            "/model/backbone/stage1/stage1.2/layers/layers.0/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage1/stage1.2/layers/layers.0/cross_resolution_weighting/Mul_2",
            "/model/backbone/stage1/stage1.2/layers/layers.1/cross_resolution_weighting/Mul",
            "/model/backbone/stage1/stage1.2/layers/layers.1/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage1/stage1.2/Add_1",
            "/model/backbone/stage1/stage1.2/layers/layers.1/cross_resolution_weighting/Mul_2",
            "/model/backbone/stage1/stage1.2/Add_2",
            "/model/backbone/stage1/stage1.2/Add_5",
            "/model/backbone/stage1/stage1.3/layers/layers.0/cross_resolution_weighting/Mul",
            "/model/backbone/stage1/stage1.3/layers/layers.0/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage1/stage1.3/layers/layers.0/cross_resolution_weighting/Mul_2",
            "/model/backbone/stage1/stage1.3/layers/layers.1/cross_resolution_weighting/Mul",
            "/model/backbone/stage1/stage1.3/layers/layers.1/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage1/stage1.3/Add_1",
            "/model/backbone/stage1/stage1.3/layers/layers.1/cross_resolution_weighting/Mul_2",
            "/model/backbone/stage1/stage1.3/Add_2",
            "/model/backbone/stage1/stage1.3/Add_5",
            "/model/aggregator/Add",
            "/model/aggregator/Add_1",
        ]

        return {
            "ignored_scope": {
                "names": ignored_scope_names,
            },
            "preset": "mixed",
        }


class LiteHRNet18(BaseSegmNNModel):
    """LiteHRNet18 Model."""

    default_backbone_configuration: ClassVar[dict[str, Any]] = {
        "norm_eval": False,
        "extra": {
            "stem": {
                "stem_channels": 32,
                "out_channels": 32,
                "expand_ratio": 1,
                "strides": [2, 2],
                "extra_stride": False,
                "input_norm": False,
            },
            "num_stages": 3,
            "stages_spec": {
                "num_modules": [2, 4, 2],
                "num_branches": [2, 3, 4],
                "num_blocks": [2, 2, 2],
                "module_type": ["LITE", "LITE", "LITE"],
                "with_fuse": [True, True, True],
                "reduce_ratios": [8, 8, 8],
                "num_channels": [[40, 80], [40, 80, 160], [40, 80, 160, 320]],
            },
            "out_modules": {
                "conv": {"enable": False, "channels": 320},
                "position_att": {"enable": False, "key_channels": 128, "value_channels": 320, "psp_size": [1, 3, 6, 8]},
                "local_att": {"enable": False},
            },
            "out_aggregator": {"enable": False},
            "add_input": False,
        },
        "pretrained_weights": "https://storage.openvinotoolkit.org/repositories/openvino_training_extensions/models/custom_semantic_segmentation/litehrnet18_imagenet1k_rsc.pth",
    }
    default_decode_head_configuration: ClassVar[dict[str, Any]] = {
        "norm_cfg": {"type": "BN", "requires_grad": True},
        "in_channels": [40, 80, 160, 320],
        "in_index": [0, 1, 2, 3],
        "input_transform": "multiple_select",
        "channels": 40,
        "enable_aggregator": True,
        "kernel_size": 1,
        "num_convs": 1,
        "concat_input": False,
        "dropout_ratio": -1,
        "align_corners": False,
    }

    @property
    def ignore_scope(self) -> dict[str, str | dict[str, list[str]]]:
        """The ignored scope of the LiteHRNet18 model."""
        ignored_scope_names = [
            "/model/backbone/stage0/stage0.0/layers/layers.0/cross_resolution_weighting/Mul",
            "/model/backbone/stage0/stage0.0/layers/layers.0/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage0/stage0.0/layers/layers.1/cross_resolution_weighting/Mul",
            "/model/backbone/stage0/stage0.0/layers/layers.1/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage0/stage0.0/Add_1",
            "/model/backbone/stage0/stage0.1/layers/layers.0/cross_resolution_weighting/Mul",
            "/model/backbone/stage0/stage0.1/layers/layers.0/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage0/stage0.1/layers/layers.1/cross_resolution_weighting/Mul",
            "/model/backbone/stage0/stage0.1/layers/layers.1/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage0/stage0.1/Add_1",
            "/model/backbone/stage1/stage1.0/layers/layers.0/cross_resolution_weighting/Mul",
            "/model/backbone/stage1/stage1.0/layers/layers.0/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage1/stage1.0/layers/layers.0/cross_resolution_weighting/Mul_2",
            "/model/backbone/stage1/stage1.0/layers/layers.1/cross_resolution_weighting/Mul",
            "/model/backbone/stage1/stage1.0/layers/layers.1/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage1/stage1.0/Add_1",
            "/model/backbone/stage1/stage1.0/layers/layers.1/cross_resolution_weighting/Mul_2",
            "/model/backbone/stage1/stage1.0/Add_2",
            "/model/backbone/stage1/stage1.0/Add_5",
            "/model/backbone/stage1/stage1.1/layers/layers.0/cross_resolution_weighting/Mul",
            "/model/backbone/stage1/stage1.1/layers/layers.0/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage1/stage1.1/layers/layers.0/cross_resolution_weighting/Mul_2",
            "/model/backbone/stage1/stage1.1/layers/layers.1/cross_resolution_weighting/Mul",
            "/model/backbone/stage1/stage1.1/layers/layers.1/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage1/stage1.1/Add_1",
            "/model/backbone/stage1/stage1.1/layers/layers.1/cross_resolution_weighting/Mul_2",
            "/model/backbone/stage1/stage1.1/Add_2",
            "/model/backbone/stage1/stage1.1/Add_5",
            "/model/backbone/stage1/stage1.2/layers/layers.0/cross_resolution_weighting/Mul",
            "/model/backbone/stage1/stage1.2/layers/layers.0/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage1/stage1.2/layers/layers.0/cross_resolution_weighting/Mul_2",
            "/model/backbone/stage1/stage1.2/layers/layers.1/cross_resolution_weighting/Mul",
            "/model/backbone/stage1/stage1.2/layers/layers.1/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage1/stage1.2/Add_1",
            "/model/backbone/stage1/stage1.2/layers/layers.1/cross_resolution_weighting/Mul_2",
            "/model/backbone/stage1/stage1.2/Add_2",
            "/model/backbone/stage1/stage1.2/Add_5",
            "/model/backbone/stage1/stage1.3/layers/layers.0/cross_resolution_weighting/Mul",
            "/model/backbone/stage1/stage1.3/layers/layers.0/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage1/stage1.3/layers/layers.0/cross_resolution_weighting/Mul_2",
            "/model/backbone/stage1/stage1.3/layers/layers.1/cross_resolution_weighting/Mul",
            "/model/backbone/stage1/stage1.3/layers/layers.1/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage1/stage1.3/Add_1",
            "/model/backbone/stage1/stage1.3/layers/layers.1/cross_resolution_weighting/Mul_2",
            "/model/backbone/stage1/stage1.3/Add_2",
            "/model/backbone/stage1/stage1.3/Add_5",
            "/model/backbone/stage2/stage2.0/layers/layers.0/cross_resolution_weighting/Mul",
            "/model/backbone/stage2/stage2.0/layers/layers.0/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage2/stage2.0/layers/layers.0/cross_resolution_weighting/Mul_2",
            "/model/backbone/stage2/stage2.0/layers/layers.0/cross_resolution_weighting/Mul_3",
            "/model/backbone/stage2/stage2.0/layers/layers.1/cross_resolution_weighting/Mul",
            "/model/backbone/stage2/stage2.0/layers/layers.1/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage2/stage2.0/Add_1",
            "/model/backbone/stage2/stage2.0/layers/layers.1/cross_resolution_weighting/Mul_2",
            "/model/backbone/stage2/stage2.0/Add_2",
            "/model/backbone/stage2/stage2.0/layers/layers.1/cross_resolution_weighting/Mul_3",
            "/model/backbone/stage2/stage2.0/Add_3",
            "/model/backbone/stage2/stage2.0/Add_6",
            "/model/backbone/stage2/stage2.0/Add_7",
            "/model/backbone/stage2/stage2.0/Add_11",
            "/model/backbone/stage2/stage2.1/layers/layers.0/cross_resolution_weighting/Mul",
            "/model/backbone/stage2/stage2.1/layers/layers.0/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage2/stage2.1/layers/layers.0/cross_resolution_weighting/Mul_2",
            "/model/backbone/stage2/stage2.1/layers/layers.0/cross_resolution_weighting/Mul_3",
            "/model/backbone/stage2/stage2.1/layers/layers.1/cross_resolution_weighting/Mul",
            "/model/backbone/stage2/stage2.1/layers/layers.1/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage2/stage2.1/Add_1",
            "/model/backbone/stage2/stage2.1/layers/layers.1/cross_resolution_weighting/Mul_2",
            "/model/backbone/stage2/stage2.1/Add_2",
            "/model/backbone/stage2/stage2.1/layers/layers.1/cross_resolution_weighting/Mul_3",
            "/model/backbone/stage2/stage2.1/Add_3",
            "/model/backbone/stage2/stage2.1/Add_6",
            "/model/backbone/stage2/stage2.1/Add_7",
            "/model/backbone/stage2/stage2.1/Add_11",
            "/model/aggregator/Add",
            "/model/aggregator/Add_1",
            "/model/aggregator/Add_2",
            "/model/backbone/stage2/stage2.1/Add",
        ]

        return {
            "ignored_scope": {
                "patterns": ["/model/backbone/*"],
                "names": ignored_scope_names,
            },
            "preset": "mixed",
        }


class LiteHRNetX(BaseSegmNNModel):
    """LiteHRNetX Model."""

    default_backbone_configuration: ClassVar[dict[str, Any]] = {
        "norm_cfg": {"type": "BN", "requires_grad": True},
        "norm_eval": False,
        "extra": {
            "stem": {
                "stem_channels": 60,
                "out_channels": 60,
                "expand_ratio": 1,
                "strides": [2, 1],
                "extra_stride": False,
                "input_norm": False,
            },
            "num_stages": 4,
            "stages_spec": {
                "weighting_module_version": "v1",
                "num_modules": [2, 4, 4, 2],
                "num_branches": [2, 3, 4, 5],
                "num_blocks": [2, 2, 2, 2],
                "module_type": ["LITE", "LITE", "LITE", "LITE"],
                "with_fuse": [True, True, True, True],
                "reduce_ratios": [2, 4, 8, 8],
                "num_channels": [[18, 60], [18, 60, 80], [18, 60, 80, 160], [18, 60, 80, 160, 320]],
            },
            "out_modules": {
                "conv": {"enable": False, "channels": 320},
                "position_att": {"enable": False, "key_channels": 128, "value_channels": 320, "psp_size": [1, 3, 6, 8]},
                "local_att": {"enable": False},
            },
            "out_aggregator": {"enable": False},
            "add_input": False,
        },
        "pretrained_weights": "https://storage.openvinotoolkit.org/repositories/openvino_training_extensions/models/custom_semantic_segmentation/litehrnetxv3_imagenet1k_rsc.pth",
    }
    default_decode_head_configuration: ClassVar[dict[str, Any]] = {
        "norm_cfg": {"type": "BN", "requires_grad": True},
        "in_channels": [18, 60, 80, 160, 320],
        "in_index": [0, 1, 2, 3, 4],
        "input_transform": "multiple_select",
        "channels": 60,
        "kernel_size": 1,
        "num_convs": 1,
        "concat_input": False,
        "dropout_ratio": -1,
        "enable_aggregator": True,
        "aggregator_min_channels": 60,
        "aggregator_merge_norm": "None",
        "aggregator_use_concat": False,
        "align_corners": False,
    }

    @property
    def ignore_scope(self) -> dict[str, str | dict[str, list[str]]]:
        """The ignored scope of the LiteHRNetX model."""
        ignored_scope_names = [
            "/model/backbone/stage0/stage0.0/layers/layers.0/cross_resolution_weighting/Mul",
            "/model/backbone/stage0/stage0.0/layers/layers.0/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage0/stage0.0/layers/layers.1/cross_resolution_weighting/Mul",
            "/model/backbone/stage0/stage0.0/layers/layers.1/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage0/stage0.0/Add_1",
            "/model/backbone/stage0/stage0.1/layers/layers.0/cross_resolution_weighting/Mul",
            "/model/backbone/stage0/stage0.1/layers/layers.0/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage0/stage0.1/layers/layers.1/cross_resolution_weighting/Mul",
            "/model/backbone/stage0/stage0.1/layers/layers.1/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage0/stage0.1/Add_1",
            "/model/backbone/stage1/stage1.0/layers/layers.0/cross_resolution_weighting/Mul",
            "/model/backbone/stage1/stage1.0/layers/layers.0/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage1/stage1.0/layers/layers.0/cross_resolution_weighting/Mul_2",
            "/model/backbone/stage1/stage1.0/layers/layers.1/cross_resolution_weighting/Mul",
            "/model/backbone/stage1/stage1.0/layers/layers.1/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage1/stage1.0/Add_1",
            "/model/backbone/stage1/stage1.0/layers/layers.1/cross_resolution_weighting/Mul_2",
            "/model/backbone/stage1/stage1.0/Add_2",
            "/model/backbone/stage1/stage1.0/Add_5",
            "/model/backbone/stage1/stage1.1/layers/layers.0/cross_resolution_weighting/Mul",
            "/model/backbone/stage1/stage1.1/layers/layers.0/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage1/stage1.1/layers/layers.0/cross_resolution_weighting/Mul_2",
            "/model/backbone/stage1/stage1.1/layers/layers.1/cross_resolution_weighting/Mul",
            "/model/backbone/stage1/stage1.1/layers/layers.1/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage1/stage1.1/Add_1",
            "/model/backbone/stage1/stage1.1/layers/layers.1/cross_resolution_weighting/Mul_2",
            "/model/backbone/stage1/stage1.1/Add_2",
            "/model/backbone/stage1/stage1.1/Add_5",
            "/model/backbone/stage1/stage1.2/layers/layers.0/cross_resolution_weighting/Mul",
            "/model/backbone/stage1/stage1.2/layers/layers.0/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage1/stage1.2/layers/layers.0/cross_resolution_weighting/Mul_2",
            "/model/backbone/stage1/stage1.2/layers/layers.1/cross_resolution_weighting/Mul",
            "/model/backbone/stage1/stage1.2/layers/layers.1/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage1/stage1.2/Add_1",
            "/model/backbone/stage1/stage1.2/layers/layers.1/cross_resolution_weighting/Mul_2",
            "/model/backbone/stage1/stage1.2/Add_2",
            "/model/backbone/stage1/stage1.2/Add_5",
            "/model/backbone/stage1/stage1.3/layers/layers.0/cross_resolution_weighting/Mul",
            "/model/backbone/stage1/stage1.3/layers/layers.0/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage1/stage1.3/layers/layers.0/cross_resolution_weighting/Mul_2",
            "/model/backbone/stage1/stage1.3/layers/layers.1/cross_resolution_weighting/Mul",
            "/model/backbone/stage1/stage1.3/layers/layers.1/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage1/stage1.3/Add_1",
            "/model/backbone/stage1/stage1.3/layers/layers.1/cross_resolution_weighting/Mul_2",
            "/model/backbone/stage1/stage1.3/Add_2",
            "/model/backbone/stage1/stage1.3/Add_5",
            "/model/backbone/stage2/stage2.0/layers/layers.0/cross_resolution_weighting/Mul",
            "/model/backbone/stage2/stage2.0/layers/layers.0/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage2/stage2.0/layers/layers.0/cross_resolution_weighting/Mul_2",
            "/model/backbone/stage2/stage2.0/layers/layers.0/cross_resolution_weighting/Mul_3",
            "/model/backbone/stage2/stage2.0/layers/layers.1/cross_resolution_weighting/Mul",
            "/model/backbone/stage2/stage2.0/layers/layers.1/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage2/stage2.0/Add_1",
            "/model/backbone/stage2/stage2.0/layers/layers.1/cross_resolution_weighting/Mul_2",
            "/model/backbone/stage2/stage2.0/Add_2",
            "/model/backbone/stage2/stage2.0/layers/layers.1/cross_resolution_weighting/Mul_3",
            "/model/backbone/stage2/stage2.0/Add_3",
            "/model/backbone/stage2/stage2.0/Add_6",
            "/model/backbone/stage2/stage2.0/Add_7",
            "/model/backbone/stage2/stage2.0/Add_11",
            "/model/backbone/stage2/stage2.1/layers/layers.0/cross_resolution_weighting/Mul",
            "/model/backbone/stage2/stage2.1/layers/layers.0/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage2/stage2.1/layers/layers.0/cross_resolution_weighting/Mul_2",
            "/model/backbone/stage2/stage2.1/layers/layers.0/cross_resolution_weighting/Mul_3",
            "/model/backbone/stage2/stage2.1/layers/layers.1/cross_resolution_weighting/Mul",
            "/model/backbone/stage2/stage2.1/layers/layers.1/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage2/stage2.1/Add_1",
            "/model/backbone/stage2/stage2.1/layers/layers.1/cross_resolution_weighting/Mul_2",
            "/model/backbone/stage2/stage2.1/Add_2",
            "/model/backbone/stage2/stage2.1/layers/layers.1/cross_resolution_weighting/Mul_3",
            "/model/backbone/stage2/stage2.1/Add_3",
            "/model/backbone/stage2/stage2.1/Add_6",
            "/model/backbone/stage2/stage2.1/Add_7",
            "/model/backbone/stage2/stage2.1/Add_11",
            "/model/backbone/stage2/stage2.2/layers/layers.0/cross_resolution_weighting/Mul",
            "/model/backbone/stage2/stage2.2/layers/layers.0/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage2/stage2.2/layers/layers.0/cross_resolution_weighting/Mul_2",
            "/model/backbone/stage2/stage2.2/layers/layers.0/cross_resolution_weighting/Mul_3",
            "/model/backbone/stage2/stage2.2/layers/layers.1/cross_resolution_weighting/Mul",
            "/model/backbone/stage2/stage2.2/layers/layers.1/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage2/stage2.2/Add_1",
            "/model/backbone/stage2/stage2.2/layers/layers.1/cross_resolution_weighting/Mul_2",
            "/model/backbone/stage2/stage2.2/Add_2",
            "/model/backbone/stage2/stage2.2/layers/layers.1/cross_resolution_weighting/Mul_3",
            "/model/backbone/stage2/stage2.2/Add_3",
            "/model/backbone/stage2/stage2.2/Add_6",
            "/model/backbone/stage2/stage2.2/Add_7",
            "/model/backbone/stage2/stage2.2/Add_11",
            "/model/backbone/stage2/stage2.3/layers/layers.0/cross_resolution_weighting/Mul",
            "/model/backbone/stage2/stage2.3/layers/layers.0/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage2/stage2.3/layers/layers.0/cross_resolution_weighting/Mul_2",
            "/model/backbone/stage2/stage2.3/layers/layers.0/cross_resolution_weighting/Mul_3",
            "/model/backbone/stage2/stage2.3/layers/layers.1/cross_resolution_weighting/Mul",
            "/model/backbone/stage2/stage2.3/layers/layers.1/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage2/stage2.3/Add_1",
            "/model/backbone/stage2/stage2.3/layers/layers.1/cross_resolution_weighting/Mul_2",
            "/model/backbone/stage2/stage2.3/Add_2",
            "/model/backbone/stage2/stage2.3/layers/layers.1/cross_resolution_weighting/Mul_3",
            "/model/backbone/stage2/stage2.3/Add_3",
            "/model/backbone/stage2/stage2.3/Add_6",
            "/model/backbone/stage2/stage2.3/Add_7",
            "/model/backbone/stage2/stage2.3/Add_11",
            "/model/backbone/stage3/stage3.0/layers/layers.0/cross_resolution_weighting/Mul",
            "/model/backbone/stage3/stage3.0/layers/layers.0/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage3/stage3.0/layers/layers.0/cross_resolution_weighting/Mul_2",
            "/model/backbone/stage3/stage3.0/layers/layers.0/cross_resolution_weighting/Mul_3",
            "/model/backbone/stage3/stage3.0/layers/layers.0/cross_resolution_weighting/Mul_4",
            "/model/backbone/stage3/stage3.0/layers/layers.1/cross_resolution_weighting/Mul",
            "/model/backbone/stage3/stage3.0/layers/layers.1/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage3/stage3.0/Add_1",
            "/model/backbone/stage3/stage3.0/layers/layers.1/cross_resolution_weighting/Mul_2",
            "/model/backbone/stage3/stage3.0/Add_2",
            "/model/backbone/stage3/stage3.0/layers/layers.1/cross_resolution_weighting/Mul_3",
            "/model/backbone/stage3/stage3.0/Add_3",
            "/model/backbone/stage3/stage3.0/layers/layers.1/cross_resolution_weighting/Mul_4",
            "/model/backbone/stage3/stage3.0/Add_4",
            "/model/backbone/stage3/stage3.0/Add_7",
            "/model/backbone/stage3/stage3.0/Add_8",
            "/model/backbone/stage3/stage3.0/Add_9",
            "/model/backbone/stage3/stage3.0/Add_13",
            "/model/backbone/stage3/stage3.0/Add_14",
            "/model/backbone/stage3/stage3.0/Add_19",
            "/model/backbone/stage3/stage3.1/layers/layers.0/cross_resolution_weighting/Mul",
            "/model/backbone/stage3/stage3.1/layers/layers.0/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage3/stage3.1/layers/layers.0/cross_resolution_weighting/Mul_2",
            "/model/backbone/stage3/stage3.1/layers/layers.0/cross_resolution_weighting/Mul_3",
            "/model/backbone/stage3/stage3.1/layers/layers.0/cross_resolution_weighting/Mul_4",
            "/model/backbone/stage3/stage3.1/layers/layers.1/cross_resolution_weighting/Mul",
            "/model/backbone/stage3/stage3.1/layers/layers.1/cross_resolution_weighting/Mul_1",
            "/model/backbone/stage3/stage3.1/Add_1",
            "/model/backbone/stage3/stage3.1/layers/layers.1/cross_resolution_weighting/Mul_2",
            "/model/backbone/stage3/stage3.1/Add_2",
            "/model/backbone/stage3/stage3.1/layers/layers.1/cross_resolution_weighting/Mul_3",
            "/model/backbone/stage3/stage3.1/Add_3",
            "/model/backbone/stage3/stage3.1/layers/layers.1/cross_resolution_weighting/Mul_4",
            "/model/backbone/stage3/stage3.1/Add_4",
            "/model/backbone/stage3/stage3.1/Add_7",
            "/model/backbone/stage3/stage3.1/Add_8",
            "/model/backbone/stage3/stage3.1/Add_9",
            "/model/backbone/stage3/stage3.1/Add_13",
            "/model/backbone/stage3/stage3.1/Add_14",
            "/model/backbone/stage3/stage3.1/Add_19",
            "/model/backbone/stage0/stage0.0/Add",
            "/model/backbone/stage0/stage0.1/Add",
            "/model/backbone/stage1/stage1.0/Add",
            "/model/backbone/stage1/stage1.1/Add",
            "/model/backbone/stage1/stage1.2/Add",
            "/model/backbone/stage1/stage1.3/Add",
            "/model/backbone/stage2/stage2.0/Add",
            "/model/backbone/stage2/stage2.1/Add",
            "/model/backbone/stage2/stage2.2/Add",
            "/model/backbone/stage2/stage2.3/Add",
            "/model/backbone/stage3/stage3.0/Add",
            "/model/backbone/stage3/stage3.1/Add",
        ]

        return {
            "ignored_scope": {
                "patterns": ["/model/aggregator/*"],
                "names": ignored_scope_names,
            },
            "preset": "performance",
        }


LITEHRNET_VARIANTS = {
    "LiteHRNet18": LiteHRNet18,
    "LiteHRNetS": LiteHRNetS,
    "LiteHRNetX": LiteHRNetX,
}


class OTXLiteHRNet(TorchVisionCompatibleModel):
    """LiteHRNet Model."""

    def _create_model(self) -> nn.Module:
        litehrnet_model_class = LITEHRNET_VARIANTS[self.name_base_model]
        # merge configurations with defaults overriding them
        backbone_configuration = self.backbone_configuration | litehrnet_model_class.default_backbone_configuration
        decode_head_configuration = (
            self.decode_head_configuration | litehrnet_model_class.default_decode_head_configuration
        )
        # initialize backbones
        backbone = LiteHRNet(**backbone_configuration)
        decode_head = CustomFCNHead(num_classes=self.num_classes, **decode_head_configuration)

        return litehrnet_model_class(
            backbone=backbone,
            decode_head=decode_head,
            criterion_configuration=self.criterion_configuration,
        )

    def load_from_otx_v1_ckpt(self, state_dict: dict, add_prefix: str = "model.model.") -> dict:
        """Load the previous OTX ckpt according to OTX2.0."""
        return OTXv1Helper.load_seg_lite_hrnet_ckpt(state_dict, add_prefix)

    @property
    def _optimization_config(self) -> dict[str, Any]:
        """PTQ config for LiteHRNet."""
        # TODO(Kirill): check PTQ without adding the whole backbone to ignored_scope
        ignored_scope = self.model.ignore_scope
        optim_config = {
            "advanced_parameters": {
                "activations_range_estimator_params": {
                    "min": {"statistics_type": "QUANTILE", "aggregator_type": "MIN", "quantile_outlier_prob": 1e-4},
                    "max": {"statistics_type": "QUANTILE", "aggregator_type": "MAX", "quantile_outlier_prob": 1e-4},
                },
            },
        }
        optim_config.update(ignored_scope)
        return optim_config
