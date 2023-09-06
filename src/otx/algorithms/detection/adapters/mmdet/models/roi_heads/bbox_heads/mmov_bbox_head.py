"""MMOV bbox head for OTX."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from copy import deepcopy
from typing import Dict, List, Optional, Union

import openvino.runtime as ov
import torch
from mmdet.models.builder import HEADS
from mmdet.models.roi_heads.bbox_heads.bbox_head import BBoxHead

from otx.core.ov.models.mmov_model import MMOVModel

# TODO: Need to fix pylint issues
# pylint: disable=too-many-instance-attributes, too-many-arguments, keyword-arg-before-vararg, dangerous-default-value


@HEADS.register_module()
class MMOVBBoxHead(BBoxHead):
    """MMOVBBoxHead class for OTX."""

    def __init__(
        self,
        model_path_or_model: Union[str, ov.Model],
        weight_path: Optional[str] = None,
        inputs: Dict[str, Union[str, List[str]]] = {},
        outputs: Dict[str, Union[str, List[str]]] = {},
        init_weight: bool = False,
        verify_shape: bool = True,
        background_index: Optional[int] = None,
        *args,
        **kwargs,
    ):
        self._model_path_or_model = model_path_or_model
        self._weight_path = weight_path
        self._inputs = deepcopy(inputs)
        self._outputs = deepcopy(outputs)
        self._init_weight = init_weight
        self._verify_sahpe = verify_shape
        self._background_index = background_index
        super().__init__(*args, **kwargs)

        if self._background_index is not None and self._background_index < 0:
            self._background_index = self.num_classes + 1 - self._background_index

        if "extractor" in inputs and "extractor" in outputs:
            self.extractor = MMOVModel(
                self._model_path_or_model,
                inputs=inputs["extractor"],
                outputs=outputs["extractor"],
                remove_normalize=False,
                merge_bn=True,
                paired_bn=True,
                init_weight=self._init_weight,
                verify_shape=self._verify_sahpe,
            )

        if self.with_cls:
            assert "fc_cls" in inputs and "fc_cls" in outputs
            self.fc_cls = MMOVModel(
                self._model_path_or_model,
                inputs=inputs["fc_cls"],
                outputs=outputs["fc_cls"],
                remove_normalize=False,
                merge_bn=False,
                paired_bn=False,
                init_weight=self._init_weight,
                verify_shape=self._verify_sahpe,
            )

        if self.with_reg:
            assert "fc_reg" in inputs and "fc_reg" in outputs
            self.fc_reg = MMOVModel(
                self._model_path_or_model,
                inputs=inputs["fc_reg"],
                outputs=outputs["fc_reg"],
                remove_normalize=False,
                merge_bn=False,
                paired_bn=False,
                init_weight=self._init_weight,
                verify_shape=self._verify_sahpe,
            )

    def init_weights(self):
        """Initialize weights of MMOVBBoxHead."""
        # TODO
        return

    def forward(self, x):
        """Forward function of MMOVBBoxHead."""
        if getattr(self, "extractor"):
            x = self.extractor(x)
        cls_score = self.fc_cls(x) if self.with_cls else None
        bbox_pred = self.fc_reg(x) if self.with_reg else None

        # since mmdet v2.0, BBoxHead is supposed to be
        # that FG labels to [0, num_class-1] and BG labels to num_class
        # but faster_rcnn_resnet50_coco, etc. from OMZ are
        # that FG labels to be [1, num_class] and BG labels to be 0
        if (
            self._background_index is not None
            and cls_score is not None
            and self._background_index != cls_score.shape(-1)
        ):
            cls_score = torch.cat(
                (
                    cls_score[:, : self._background_index],
                    cls_score[:, self._background_index + 1 :],
                    cls_score[:, self._background_index : self._background_index + 1],
                ),
                -1,
            )

        return (cls_score, bbox_pred)
