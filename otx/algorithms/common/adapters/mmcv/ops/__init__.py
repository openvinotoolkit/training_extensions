"""Initial file for mmcv ops."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from mmcv.ops import multi_scale_deform_attn

from .multi_scale_deformable_attn_pytorch import multi_scale_deformable_attn_pytorch

multi_scale_deform_attn.multi_scale_deformable_attn_pytorch = multi_scale_deformable_attn_pytorch
