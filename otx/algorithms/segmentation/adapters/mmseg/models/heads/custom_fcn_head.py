"""Custom FCN head."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from mmcv.runner import force_fp32
from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.fcn_head import FCNHead

from otx.algorithms.segmentation.adapters.mmseg.models.utils import IterativeAggregator

@HEADS.register_module()
class CustomFCNHead(FCNHead):  # pylint: disable=too-many-ancestors
    """Custom Fully Convolution Networks for Semantic Segmentation."""

    def __init__(self,
        enable_aggregator=False,
        aggregator_min_channels=None,
        aggregator_merge_norm=None,
        aggregator_use_concat=False,
        *args, **kwargs):

        in_channels = kwargs.get("in_channels")
        in_index = kwargs.get("in_index")
        norm_cfg = kwargs.get("norm_cfg")
        conv_cfg = kwargs.get("conv_cfg")
        input_transform = kwargs.get("input_transform")

        aggregator = None
        if enable_aggregator or input_transform == "multiple_select":
            assert isinstance(in_channels, (tuple, list))
            assert len(in_channels) > 1

            aggregator = IterativeAggregator(
                in_channels=in_channels,
                min_channels=aggregator_min_channels,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                merge_norm=aggregator_merge_norm,
                use_concat=aggregator_use_concat,
            )

            aggregator_min_channels = aggregator_min_channels if aggregator_min_channels is not None else 0
            # change arguments temporarily
            kwargs["in_channels"] = max(in_channels[0], aggregator_min_channels)
            kwargs["input_transform"] = None
            if in_index is not None:
                kwargs["in_index"] = in_index[0]

        super().__init__(*args, **kwargs)

        self.aggregator = aggregator

        # re-define variables
        self.in_channels = in_channels
        self.input_transform = input_transform
        self.in_index = in_index

    def _transform_inputs(self, inputs):
        inputs = super()._transform_inputs(inputs)
        if self.aggregator is not None:
            inputs = self.aggregator(inputs)[0]
        return inputs
