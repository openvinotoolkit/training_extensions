"""MMOV YOLOX Head for OTX."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from copy import deepcopy
from typing import Dict, List, Optional, Union

import openvino.runtime as ov
import torch
from mmdet.models.builder import HEADS
from mmdet.models.dense_heads.yolo_head import YOLOV3Head

from otx.core.ov.models.mmov_model import MMOVModel

# TODO: Need to fix pylint issues
# pylint: disable=too-many-instance-attributes, keyword-arg-before-vararg


@HEADS.register_module()
class MMOVYOLOV3Head(YOLOV3Head):
    """MMOVYOLOV3Head class for OTX."""

    def __init__(
        self,
        model_path_or_model: Union[str, ov.Model],
        weight_path: Optional[str] = None,
        inputs: Optional[Union[Dict[str, Union[str, List[str]]], List[str], str]] = None,
        outputs: Optional[Union[Dict[str, Union[str, List[str]]], List[str], str]] = None,
        init_weight: bool = False,
        verify_shape: bool = True,
        *args,
        **kwargs,
    ):
        self._model_path_or_model = model_path_or_model
        self._weight_path = weight_path
        self._inputs = deepcopy(inputs)
        self._outputs = deepcopy(outputs)
        self._init_weight = init_weight
        self._verify_shape = verify_shape

        # dummy input
        in_channels = (512, 256, 128)
        out_channels = (1024, 512, 256)
        if "featmap_strides" in kwargs:
            in_channels = kwargs["featmap_strides"]
            out_channels = kwargs["featmap_strides"]
        super().__init__(in_channels=in_channels, out_channels=out_channels, *args, **kwargs)

    def _init_layers(self):
        """Initialize layers of MMOVModels."""
        self.convs_bridge = torch.nn.ModuleList()
        self.convs_pred = torch.nn.ModuleList()

        for inputs, outputs in zip(self._inputs["convs_bridge"], self._outputs["convs_bridge"]):
            self.convs_bridge.append(
                MMOVModel(
                    self._model_path_or_model,
                    self._weight_path,
                    inputs=inputs,
                    outputs=outputs,
                    remove_normalize=False,
                    init_weight=self._init_weight,
                    verify_shape=self._verify_shape,
                )
            )

        for inputs, outputs in zip(self._inputs["convs_pred"], self._outputs["convs_pred"]):
            self.convs_pred.append(
                MMOVModel(
                    self._model_path_or_model,
                    self._weight_path,
                    inputs=inputs,
                    outputs=outputs,
                    remove_normalize=False,
                    init_weight=self._init_weight,
                    verify_shape=self._verify_shape,
                )
            )

    def init_weights(self):
        """Initialize weights of MMOVYOLOV3Head."""
        # TODO
        return
