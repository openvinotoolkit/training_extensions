# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.
"""Implementation modified from mmdet.models.roi_heads.bbox_heads.convfc_bbox_head.py.

Reference : https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/models/roi_heads/bbox_heads/convfc_bbox_head.py
"""

from __future__ import annotations

from typing import Callable

from torch import Tensor, nn

from .bbox_head import BBoxHead


class ConvFCBBoxHead(BBoxHead):
    r"""More general bbox head, with shared conv and fc layers and two optional separated branches.

    .. code-block:: none

                                    /-> cls convs -> cls fcs -> cls
        shared convs -> shared fcs
                                    \-> reg convs -> reg fcs -> reg
    """

    def __init__(
        self,
        num_shared_convs: int = 0,
        num_shared_fcs: int = 0,
        num_cls_convs: int = 0,
        num_cls_fcs: int = 0,
        num_reg_convs: int = 0,
        num_reg_fcs: int = 0,
        conv_out_channels: int = 256,
        fc_out_channels: int = 1024,
        normalization: Callable[..., nn.Module] | None = None,
        init_cfg: dict | None = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, init_cfg=init_cfg, **kwargs)  # type: ignore [misc]
        if num_shared_convs + num_shared_fcs + num_cls_convs + num_cls_fcs + num_reg_convs + num_reg_fcs <= 0:
            msg = (
                "Pls specify at least one of num_shared_convs, num_shared_fcs, num_cls_convs, num_cls_fcs, "
                "num_reg_convs, num_reg_fcs"
            )
            raise ValueError(msg)
        if (num_cls_convs > 0 or num_reg_convs > 0) and num_shared_fcs != 0:
            msg = "Shared FC layers are mutually exclusive with cls/reg conv layers"
            raise ValueError(msg)
        if (not self.with_cls) and (num_cls_convs != 0 or num_cls_fcs != 0):
            msg = "num_cls_convs and num_cls_fcs should be zero if without classification"
            raise ValueError(msg)
        if (not self.with_reg) and (num_reg_convs != 0 or num_reg_fcs != 0):
            msg = "num_reg_convs and num_reg_fcs should be zero if without regression"
            raise ValueError(msg)
        self.num_shared_convs = num_shared_convs
        self.num_shared_fcs = num_shared_fcs
        self.num_cls_convs = num_cls_convs
        self.num_cls_fcs = num_cls_fcs
        self.num_reg_convs = num_reg_convs
        self.num_reg_fcs = num_reg_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.normalization = normalization

        # add shared convs and fcs
        self.shared_convs, self.shared_fcs, last_layer_dim = self._add_conv_fc_branch(
            self.num_shared_convs,
            self.num_shared_fcs,
            self.in_channels,
            True,
        )
        self.shared_out_channels = last_layer_dim

        # add cls specific branch
        self.cls_convs, self.cls_fcs, self.cls_last_dim = self._add_conv_fc_branch(
            self.num_cls_convs,
            self.num_cls_fcs,
            self.shared_out_channels,
        )

        # add reg specific branch
        self.reg_convs, self.reg_fcs, self.reg_last_dim = self._add_conv_fc_branch(
            self.num_reg_convs,
            self.num_reg_fcs,
            self.shared_out_channels,
        )

        self.relu = nn.ReLU(inplace=True)
        # reconstruct fc_cls and fc_reg since input channels are changed
        if self.with_cls:
            cls_channels = self.num_classes + 1
            self.fc_cls = nn.Linear(in_features=self.cls_last_dim, out_features=cls_channels)
        if self.with_reg:
            box_dim = self.bbox_coder.encode_size
            out_dim_reg = box_dim if self.reg_class_agnostic else box_dim * self.num_classes
            self.fc_reg = nn.Linear(in_features=self.reg_last_dim, out_features=out_dim_reg)

        if init_cfg is None:
            # when init_cfg is None,
            # It has been set to
            # [[dict(type='Normal', std=0.01, override=dict(name='fc_cls'))],
            #  [dict(type='Normal', std=0.001, override=dict(name='fc_reg'))]
            # after `super(ConvFCBBoxHead, self).__init__()`
            # we only need to append additional configuration
            # for `shared_fcs`, `cls_fcs` and `reg_fcs`
            if not isinstance(self.init_cfg, list):
                msg = "init_cfg must be a list"
                raise TypeError(msg)
            self.init_cfg += [
                {
                    "type": "Xavier",
                    "distribution": "uniform",
                    "override": [
                        {"name": "shared_fcs"},
                        {"name": "cls_fcs"},
                        {"name": "reg_fcs"},
                    ],
                },
            ]

    def _add_conv_fc_branch(
        self,
        num_branch_convs: int,
        num_branch_fcs: int,
        in_channels: int,
        is_shared: bool = False,
    ) -> tuple:
        """Add shared or separable branch.

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()

        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if (is_shared or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = last_layer_dim if i == 0 else self.fc_out_channels
                branch_fcs.append(nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def forward(self, x: Tensor) -> tuple:
        """Forward features from the upstream network.

        Args:
            x (Tensor): Features from the upstream network, each is a 4D-tensor.

        Returns:
            tuple: A tuple of classification scores and bbox prediction.

                - cls_score (Tensor): Classification scores for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_base_priors * num_classes.
                - bbox_pred (Tensor): Box energies / deltas for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_base_priors * 4.
        """
        # shared part

        if self.num_shared_fcs > 0:
            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        x_cls = x
        x_reg = x

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        return cls_score, bbox_pred


class Shared2FCBBoxHead(ConvFCBBoxHead):
    """Shared 2 FC BBox Head."""

    def __init__(self, fc_out_channels: int = 1024, *args, **kwargs) -> None:
        super().__init__(  # type: ignore [misc]
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,  # noqa: B026
            **kwargs,
        )
