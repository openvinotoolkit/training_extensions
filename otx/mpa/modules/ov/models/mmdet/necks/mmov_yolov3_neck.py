# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from typing import Dict, List, Optional, Union

import torch
from mmdet.models.builder import NECKS
from mmdet.models.necks.yolo_neck import YOLOV3Neck

from ...mmov_model import MMOVModel
from ...parser_mixin import ParserMixin


@NECKS.register_module()
class MMOVYOLOV3Neck(YOLOV3Neck, ParserMixin):
    def __init__(
        self,
        model_path: str,
        weight_path: Optional[str] = None,
        inputs: Optional[Union[Dict[str, Union[str, List[str]]], List[str], str]] = None,
        outputs: Optional[Union[Dict[str, Union[str, List[str]]], List[str], str]] = None,
        init_weight: bool = False,
        verify_shape: bool = True,
    ):
        super(YOLOV3Neck, self).__init__()

        self._model_path = model_path
        self._weight_path = weight_path
        self._init_weight = init_weight

        inputs, outputs = super().parse(
            model_path=model_path,
            weight_path=weight_path,
            inputs=inputs,
            outputs=outputs,
        )

        for key in inputs.keys():
            input = inputs[key]
            output = outputs[key]

            if input and output:
                self.add_module(
                    key,
                    MMOVModel(
                        model_path,
                        weight_path,
                        inputs=input,
                        outputs=output,
                        remove_normalize=False,
                        init_weight=init_weight,
                        verify_shape=verify_shape,
                    ),
                )
            else:
                self.add_module(key, torch.nn.Identity())

        self.num_scales = len([key for key in inputs.keys() if key.startswith("detect")])

    def init_weights(self, pretrained=None):
        # TODO
        pass
