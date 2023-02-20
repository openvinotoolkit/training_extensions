# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from typing import Dict, List, Optional, Union

import openvino.runtime as ov
import torch.nn.functional as F
from mmcls.models.builder import HEADS
from mmcls.models.heads import ClsHead

from ....graph.parsers.cls import cls_base_parser
from ...mmov_model import MMOVModel


@HEADS.register_module()
class MMOVClsHead(ClsHead):
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
    ):
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

    def forward_train(self, x, gt_label, **kwargs):
        cls_score = self.model(x)
        while cls_score.dim() > 2:
            cls_score = cls_score.squeeze(2)
        losses = self.loss(cls_score, gt_label, **kwargs)
        return losses

    def simple_test(self, x):
        cls_score = self.model(x)
        while cls_score.dim() > 2:
            cls_score = cls_score.squeeze(2)
        if self._softmax_at_test:
            pred = F.softmax(cls_score, dim=1) if cls_score is not None else None
        else:
            pred = cls_score
        return self.post_process(pred)
