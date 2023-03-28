"""MMOV Mask Head for OTX."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from copy import deepcopy
from typing import Dict, List, Optional, Union

import openvino.runtime as ov
from mmdet.models.builder import HEADS
from mmdet.models.roi_heads.mask_heads.fcn_mask_head import FCNMaskHead

from otx.core.ov.models.mmov_model import MMOVModel

# TODO: Need to fix pylint issues
# pylint: disable=too-many-instance-attributes, too-many-arguments, keyword-arg-before-vararg, dangerous-default-value


@HEADS.register_module()
class MMOVMaskHead(FCNMaskHead):
    """MMOVMaskHead class for OTX."""

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

        # dummy input
        super().__init__(*args, **kwargs)
        delattr(self, "convs")
        delattr(self, "upsample")
        delattr(self, "conv_logits")
        delattr(self, "relu")

        #  if self._background_index is not None and self._background_index < 0:
        #      self._background_index = self.num_classes + 1 - self._background_index

        self.model = MMOVModel(
            self._model_path_or_model,
            inputs=inputs,
            outputs=outputs,
            remove_normalize=False,
            merge_bn=True,
            paired_bn=True,
            init_weight=self._init_weight,
            verify_shape=self._verify_sahpe,
        )

    def init_weights(self):
        """Initial weights of MMOVMaskHead."""
        # TODO
        return

    def forward(self, x):
        """Forward function of MMOVMaskHead."""
        return self.model(x)
