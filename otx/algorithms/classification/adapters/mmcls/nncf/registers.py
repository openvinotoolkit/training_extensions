"""Register custom modules for mmcls models."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from otx.algorithms.common.adapters.nncf.utils import is_nncf_enabled

if is_nncf_enabled():
    from nncf.torch import register_module
    from timm.models.layers.conv2d_same import Conv2dSame

    # Register custom modules.
    # Users of nncf should manually check every custom
    # layer with weights which should be compressed and
    # in case such layers are not wrapping by nncf,
    # wrap such custom module by yourself.
    register_module(ignored_algorithms=[])(Conv2dSame)
