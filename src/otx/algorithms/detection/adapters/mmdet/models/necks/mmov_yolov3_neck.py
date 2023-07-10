"""MMOVYOLOV3Neck class for OMZ models."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from typing import Dict, List, Optional, Union

import openvino.runtime as ov
import torch
from mmdet.models.builder import NECKS
from mmdet.models.necks.yolo_neck import YOLOV3Neck

from otx.core.ov.models.mmov_model import MMOVModel
from otx.core.ov.models.parser_mixin import ParserMixin  # type: ignore[attr-defined]


@NECKS.register_module()
class MMOVYOLOV3Neck(YOLOV3Neck, ParserMixin):
    """MMOVYOLOV3Neck class for OMZ models."""

    def __init__(
        self,
        model_path_or_model: Union[str, ov.Model],
        weight_path: Optional[str] = None,
        inputs: Optional[Union[Dict[str, Union[str, List[str]]], List[str], str]] = None,
        outputs: Optional[Union[Dict[str, Union[str, List[str]]], List[str], str]] = None,
        init_weight: bool = False,
        verify_shape: bool = True,
    ):
        super(YOLOV3Neck, self).__init__()

        self._model_path_or_model = model_path_or_model
        self._weight_path = weight_path
        self._init_weight = init_weight

        inputs, outputs = super().parse(
            model_path_or_model=model_path_or_model,
            weight_path=weight_path,
            inputs=inputs,
            outputs=outputs,
        )

        # TODO: Need to fix what exactly the types of inputs and outputs are.
        if not isinstance(inputs, dict) or not isinstance(outputs, dict):
            raise ValueError("The type of inputs & outputs is invalid.")
        for key in inputs.keys():
            input_t = inputs[key]
            output_t = outputs[key]

            if input_t and output_t:
                self.add_module(
                    key,
                    MMOVModel(
                        model_path_or_model,
                        weight_path,
                        inputs=input_t,
                        outputs=output_t,
                        remove_normalize=False,
                        init_weight=init_weight,
                        verify_shape=verify_shape,
                    ),
                )
            else:
                self.add_module(key, torch.nn.Identity())

        self.num_scales = len([key for key in inputs.keys() if key.startswith("detect")])

    def init_weights(self, pretrained=None):  # pylint: disable=unused-argument
        """Initial weights of MMOVYOLOV3Neck."""
        # TODO
        return
