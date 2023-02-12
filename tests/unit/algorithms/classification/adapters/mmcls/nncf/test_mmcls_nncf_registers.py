# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import otx.algorithms.classification.adapters.mmcls.nncf.registers  # noqa: F401
from otx.algorithms.common.adapters.nncf.utils import is_nncf_enabled
from tests.test_suite.e2e_test_system import e2e_pytest_unit


@e2e_pytest_unit
def test_registers():
    if is_nncf_enabled():
        from nncf.torch.layers import UNWRAPPED_USER_MODULES
        from timm.models.layers.conv2d_same import Conv2dSame

        assert Conv2dSame in UNWRAPPED_USER_MODULES.registry_dict.values()
