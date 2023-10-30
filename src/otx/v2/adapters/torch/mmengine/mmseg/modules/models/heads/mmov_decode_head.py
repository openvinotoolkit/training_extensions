"""Decode-head used for openvino export."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from copy import deepcopy
from typing import Dict, List, Optional, Union

import openvino.runtime as ov
from mmseg.models.decode_heads.decode_head import BaseDecodeHead

from otx.core.ov.models.mmov_model import MMOVModel

# pylint: disable=too-many-instance-attributes, keyword-arg-before-vararg


class MMOVDecodeHead(BaseDecodeHead):
    """MMOVDecodeHead."""

    def __init__(
        self,
        model_path_or_model: Union[str, ov.Model] = None,
        weight_path: Optional[str] = None,
        inputs: Optional[Dict[str, Union[str, List[str]]]] = None,
        outputs: Optional[Dict[str, Union[str, List[str]]]] = None,
        init_weight: bool = False,
        verify_shape: bool = True,
        *args,
        **kwargs
    ):
        if inputs is None:
            inputs = {}
        if outputs is None:
            outputs = {}
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
        """Init weights."""
        # TODO
        return

    def forward(self, inputs):
        """Forward."""
        outputs = self._transform_inputs(inputs)
        if getattr(self, "extractor"):
            outputs = self.extractor(outputs)
        outputs = self.cls_seg(outputs)
        return outputs
