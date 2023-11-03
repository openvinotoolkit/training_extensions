"""Custom universal class incremental otx head."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
from __future__ import annotations

import torch
from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.fcn_head import FCNHead

from otx.v2.adapters.torch.mmengine.mmseg.modules.models.utils import IterativeAggregator


@HEADS.register_module()
class OTXFCNHead(FCNHead):
    """OTXFCNHead is a fully convolutional network head used in OTX.

    Args:
        enable_aggregator (bool): Whether to enable the Lite-HRNet aggregator.
        aggregator_min_channels (int, optional): Minimum number of channels for the aggregator.
        aggregator_merge_norm (str, optional): Type of normalization to use for the aggregator.
        aggregator_use_concat (bool): Whether to use concatenation for the aggregator.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        aggregator (IterativeAggregator): The Lite-HRNet aggregator.
        in_channels (int): Number of input channels.
        input_transform (dict): Input transformation.
        in_index (int): Index of input.
        ignore_index (int): Index to ignore.

    """

    def __init__(
        self,
        enable_aggregator: bool = False,
        aggregator_min_channels: int | None = None,
        aggregator_merge_norm: str | None = None,
        aggregator_use_concat: bool = False,
        *args,
        **kwargs,
    ):
        """Initializes OTXFCNHead.

        Args:
            enable_aggregator (bool): Whether to enable the Lite-HRNet aggregator.
            aggregator_min_channels (int, optional): Minimum number of channels for the aggregator.
            aggregator_merge_norm (str, optional): Type of normalization to use for the aggregator.
            aggregator_use_concat (bool): Whether to use concatenation for the aggregator.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        """
        in_channels: list[int] = kwargs.get("in_channels", [])
        in_index = kwargs.get("in_index")
        norm_cfg = kwargs.get("norm_cfg")
        conv_cfg = kwargs.get("conv_cfg")
        input_transform = kwargs.get("input_transform")

        aggregator: torch.nn.Module
        if enable_aggregator:  # Lite-HRNet aggregator
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

        self.ignore_index = 255

        # get rid of last activation of convs module
        if self.act_cfg:
            self.convs[-1].with_activation = False
            delattr(self.convs[-1], "activate")

        if kwargs.get("init_cfg", {}):
            self.init_weights()

    def _transform_inputs(self, inputs: torch.Tensor):
        """Transforms inputs.

        Args:
            inputs (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed input tensor.

        """
        if self.aggregator is not None:
            inputs = self.aggregator(inputs)[0]
        else:
            inputs = super()._transform_inputs(inputs)

        return inputs
