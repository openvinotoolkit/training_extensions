# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from typing import Dict, List, Optional, Union

import openvino.runtime as ov
import torch.nn as nn
from mmdet.models.builder import NECKS
from mmdet.models.necks.fpn import FPN

from ...mmov_model import MMOVModel


@NECKS.register_module()
class MMOVFPN(FPN):
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
        for input, output in zip(inputs["laterals"], outputs["laterals"]):
            self.lateral_convs.append(
                MMOVModel(
                    model_path_or_model,
                    weight_path,
                    inputs=input,
                    outputs=output,
                    remove_normalize=False,
                    merge_bn=False,
                    paired_bn=False,
                    init_weight=init_weight,
                    verify_shape=verify_shape,
                )
            )

        self.fpn_convs = nn.ModuleList()
        for input, output in zip(inputs["fpn"], outputs["fpn"]):
            if input and output:
                self.fpn_convs.append(
                    MMOVModel(
                        model_path_or_model,
                        weight_path,
                        inputs=input,
                        outputs=output,
                        remove_normalize=False,
                        merge_bn=True,
                        paired_bn=True,
                        init_weight=init_weight,
                        verify_shape=verify_shape,
                    )
                )
            else:
                self.fpn_convs.append(nn.Identity())

    def init_weights(self, pretrained=None):
        # TODO
        pass
