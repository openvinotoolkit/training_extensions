"""Module for OpenVINO Classification Head adopted with mmclassification."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from typing import Dict, List, Optional, Union

import openvino.runtime as ov
import torch.nn.functional as F
from mmcls.models.builder import HEADS
from mmcls.models.heads import ClsHead

from otx.core.ov.graph.parsers.cls import cls_base_parser
from otx.core.ov.models.mmov_model import MMOVModel


@HEADS.register_module()
class MMOVClsHead(ClsHead):
    """Head module for MMClassification that uses MMOV for inference.

    Args:
        model_path_or_model (Union[str, ov.Model]): Path to the ONNX model file or
            the ONNX model object.
        weight_path (Optional[str]): Path to the weight file.
        inputs (Optional[Union[Dict[str, Union[str, List[str]]], List[str], str]]):
            Input shape(s) of the ONNX model.
        outputs (Optional[Union[Dict[str, Union[str, List[str]]], List[str], str]]):
            Output name(s) of the ONNX model.
        init_weight (bool): Whether to initialize the weight from a normal
            distribution.
        verify_shape (bool): Whether to verify the input shape of the ONNX model.
        softmax_at_test (bool): Whether to apply softmax during testing.
    """

    def __init__(
        self,
        model_path_or_model: Union[str, ov.Model],
        weight_path: Optional[str] = None,
        inputs: Optional[Union[Dict[str, Union[str, List[str]]], List[str], str]] = None,
        outputs: Optional[Union[Dict[str, Union[str, List[str]]], List[str], str]] = None,
        init_weight: bool = False,
        verify_shape: bool = True,
        softmax_at_test: bool = True,
        **kwargs,
    ):  # pylint: disable=too-many-arguments
        kwargs.pop("in_channels", None)
        kwargs.pop("num_classes", None)
        super().__init__(**kwargs)

        self._model_path_or_model = model_path_or_model
        self._weight_path = weight_path
        self._init_weight = init_weight
        self._softmax_at_test = softmax_at_test

        self.model = MMOVModel(
            model_path_or_model,
            weight_path,
            inputs=inputs,
            outputs=outputs,
            remove_normalize=False,
            merge_bn=False,
            paired_bn=False,
            init_weight=init_weight,
            verify_shape=verify_shape,
            parser=cls_base_parser,
            parser_kwargs=dict(component="head"),
        )

    def forward(self, x):
        """Forward fuction of MMOVClsHead class."""
        return self.simple_test(x)

    def forward_train(self, cls_score, gt_label, **kwargs):
        """Forward_train fuction of MMOVClsHead."""
        cls_score = self.model(cls_score)
        while cls_score.dim() > 2:
            cls_score = cls_score.squeeze(2)
        losses = self.loss(cls_score, gt_label, **kwargs)
        return losses

    def simple_test(self, cls_score):
        """Test without augmentation."""
        cls_score = self.model(cls_score)
        while cls_score.dim() > 2:
            cls_score = cls_score.squeeze(2)
        if self._softmax_at_test:
            pred = F.softmax(cls_score, dim=1) if cls_score is not None else None
        else:
            pred = cls_score
        return self.post_process(pred)
