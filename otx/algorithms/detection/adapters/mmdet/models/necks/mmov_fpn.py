"""MMOV FPN of mmdetection adapters."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from typing import Dict, List, Optional, Union

import openvino.runtime as ov
from mmdet.models.builder import NECKS
from mmdet.models.necks.fpn import FPN
from torch import nn

from otx.core.ov.models.mmov_model import MMOVModel

# TODO: Need to fix pylint issues
# pylint: disable=keyword-arg-before-vararg, too-many-locals


@NECKS.register_module()
class MMOVFPN(FPN):
    """MMOVFPN class for OMZ models."""

    def __init__(
        self,
        model_path_or_model: Union[str, ov.Model],
        weight_path: Optional[str] = None,
        inputs: Optional[Union[Dict[str, Union[str, List[str]]], List[str], str]] = None,
        outputs: Optional[Union[Dict[str, Union[str, List[str]]], List[str], str]] = None,
        init_weight: bool = False,
        verify_shape: bool = True,
        *args,
        **kwargs
    ):

        # dummy
        # TODO: Need to fix what exactly the types of inputs and outputs are.
        if not isinstance(inputs, dict) or not isinstance(outputs, dict):
            raise ValueError("The type of inputs & outputs is invalid.")
        in_channels = [8 for _ in inputs["laterals"]]
        out_channels = 8
        relu_before_extra_convs = False
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            relu_before_extra_convs=relu_before_extra_convs * args,
            **kwargs
        )

        self.lateral_convs = nn.ModuleList()
        for input_laterals, output_laterals in zip(inputs["laterals"], outputs["laterals"]):
            self.lateral_convs.append(
                MMOVModel(
                    model_path_or_model,
                    weight_path,
                    inputs=input_laterals,
                    outputs=output_laterals,
                    remove_normalize=False,
                    merge_bn=False,
                    paired_bn=False,
                    init_weight=init_weight,
                    verify_shape=verify_shape,
                )
            )

        self.fpn_convs = nn.ModuleList()
        for input_fpn, output_fpn in zip(inputs["fpn"], outputs["fpn"]):
            if input_fpn and output_fpn:
                self.fpn_convs.append(
                    MMOVModel(
                        model_path_or_model,
                        weight_path,
                        inputs=input_fpn,
                        outputs=output_fpn,
                        remove_normalize=False,
                        merge_bn=True,
                        paired_bn=True,
                        init_weight=init_weight,
                        verify_shape=verify_shape,
                    )
                )
            else:
                self.fpn_convs.append(nn.Identity())

    def init_weights(self, pretrained=None):  # pylint: disable=unused-argument
        """Initial weights function of MMOVFPN."""
        # TODO
        return
