# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from copy import deepcopy
from typing import Dict, List, Optional, Union

import openvino.runtime as ov
from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead

from ...mmov_model import MMOVModel


@HEADS.register_module()
class MMOVDecodeHead(BaseDecodeHead):
    def __init__(
        self,
        model_path_or_model: Union[str, ov.Model] = None,
        weight_path: Optional[str] = None,
        inputs: Dict[str, Union[str, List[str]]] = {},
        outputs: Dict[str, Union[str, List[str]]] = {},
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
        channels = 1
        super().__init__(
            channels=channels,
            *args,
            **kwargs,
        )
        delattr(self, "channels")

        if "extractor" in inputs and "extractor" in outputs:
            self.extractor = MMOVModel(
                self._model_path_or_model,
                self._weight_path,
                inputs=inputs["extractor"],
                outputs=outputs["extractor"],
                remove_normalize=False,
                merge_bn=True,
                paired_bn=True,
                verify_shape=self._verify_shape,
                init_weight=self._init_weight,
            )

        assert "cls_seg" in inputs and "cls_seg" in outputs
        self.conv_seg = MMOVModel(
            self._model_path_or_model,
            self._weight_path,
            inputs=inputs["cls_seg"],
            outputs=outputs["cls_seg"],
            remove_normalize=False,
            merge_bn=False,
            paired_bn=False,
            verify_shape=self._verify_shape,
            init_weight=self._init_weight,
        )

    def init_weights(self):
        # TODO
        pass

    def forward(self, inputs):
        outputs = self._transform_inputs(inputs)
        if getattr(self, "extractor"):
            outputs = self.extractor(outputs)
        outputs = self.cls_seg(outputs)
        return outputs
