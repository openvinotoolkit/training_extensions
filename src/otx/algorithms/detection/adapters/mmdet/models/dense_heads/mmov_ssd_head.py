"""MMOV SSD Head for OTX."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from copy import deepcopy
from typing import Dict, List, Optional, Union

import openvino.runtime as ov
import torch
from mmdet.core import build_anchor_generator
from mmdet.models.builder import HEADS
from mmdet.models.dense_heads.ssd_head import SSDHead

from otx.core.ov.models.mmov_model import MMOVModel

# TODO: Need to fix pylint issues
# pylint: disable=redefined-argument-from-local, too-many-instance-attributes
# pylint: disable=too-many-arguments, keyword-arg-before-vararg


@HEADS.register_module()
class MMOVSSDHead(SSDHead):
    """MMOVSSDHead class for OTX."""

    def __init__(
        self,
        model_path_or_model: Union[str, ov.Model],
        weight_path: Optional[str] = None,
        inputs: Optional[Union[Dict[str, Union[str, List[str]]], List[str], str]] = None,
        outputs: Optional[Union[Dict[str, Union[str, List[str]]], List[str], str]] = None,
        init_weight: bool = False,
        verify_shape: bool = True,
        transpose_cls: bool = False,
        transpose_reg: bool = False,
        background_index: Optional[int] = None,
        *args,
        **kwargs,
    ):

        self._model_path_or_model = model_path_or_model
        self._weight_path = weight_path
        self._inputs = deepcopy(inputs)
        self._outputs = deepcopy(outputs)
        self._init_weight = init_weight
        self._verify_shape = verify_shape
        self._transpose_cls = transpose_cls
        self._transpose_reg = transpose_reg
        self._background_index = background_index

        # dummy input
        anchor_generator = build_anchor_generator(kwargs["anchor_generator"])
        num_anchors = anchor_generator.num_base_anchors
        in_channels = [256 for _ in num_anchors]
        super().__init__(in_channels=in_channels, *args, **kwargs)

        self.cls_convs = torch.nn.ModuleList()
        self.reg_convs = torch.nn.ModuleList()

        # TODO: Need to fix what exactly the types of inputs and outputs are.
        if not isinstance(self._inputs, dict) or not isinstance(self._outputs, dict):
            raise ValueError("The type of inputs & outputs is invalid.")
        for (
            inputs,
            outputs,
        ) in zip(self._inputs["cls_convs"], self._outputs["cls_convs"]):
            self.cls_convs.append(
                MMOVModel(
                    self._model_path_or_model,
                    self._weight_path,
                    inputs=inputs,
                    outputs=outputs,
                    remove_normalize=False,
                    merge_bn=False,
                    paired_bn=False,
                    init_weight=self._init_weight,
                    verify_shape=self._verify_shape,
                )
            )

        for (
            inputs,
            outputs,
        ) in zip(self._inputs["reg_convs"], self._outputs["reg_convs"]):
            self.reg_convs.append(
                MMOVModel(
                    self._model_path_or_model,
                    self._weight_path,
                    inputs=inputs,
                    outputs=outputs,
                    remove_normalize=False,
                    merge_bn=False,
                    paired_bn=False,
                    init_weight=self._init_weight,
                    verify_shape=self._verify_shape,
                )
            )

    def forward(self, feats):
        """Forward function for MMOVSSDHead."""
        cls_scores = []
        bbox_preds = []
        for feat, reg_conv, cls_conv in zip(feats, self.reg_convs, self.cls_convs):

            cls_score = cls_conv(feat)
            bbox_pred = reg_conv(feat)

            if self._transpose_cls:
                # [B, cls_out_channels * num_anchors, H, W]
                #   -> [B, num_anchors * cls_out_channels, H, W]
                shape = cls_score.shape
                cls_score = (
                    cls_score.reshape(shape[0], self.cls_out_channels, -1, *shape[2:]).transpose(1, 2).reshape(shape)
                )

            if self._transpose_reg:
                # [B, 4 * num_anchors, H, W] -> [B, num_anchors * 4, H, W]
                shape = bbox_pred.shape
                bbox_pred = bbox_pred.reshape(shape[0], 4, -1, *shape[2:]).transpose(1, 2).reshape(shape)

            # since mmdet v2.0, SSDHead is supposed to be
            # that FG labels to [0, num_class-1] and BG labels to num_class
            # but ssd300, ssd512, etc. from OMZ are
            # that FG labels to [1, num_class] and BG labels to 0
            if self._background_index is not None and cls_score is not None:
                cls_score = cls_score.permute(0, 2, 3, 1)
                shape = cls_score.shape
                cls_score = cls_score.reshape(-1, self.cls_out_channels)
                cls_score = torch.cat(
                    (
                        cls_score[:, : self._background_index],
                        cls_score[:, self._background_index + 1 :],
                        cls_score[:, self._background_index : self._background_index + 1],
                    ),
                    -1,
                )
                cls_score = cls_score.reshape(shape)
                cls_score = cls_score.permute(0, 3, 1, 2)

            cls_scores.append(cls_score)
            bbox_preds.append(bbox_pred)
        return cls_scores, bbox_preds

    def init_weights(self):
        """Initialize weights function of MMOVSSDHead."""
        # TODO
        return
