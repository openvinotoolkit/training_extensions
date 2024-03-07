"""Test for XPU optimizer hook"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from otx.algorithms.common.utils.utils import is_xpu_available


@pytest.mark.skipif(not is_xpu_available(), reason="XPU is not available")
def test_init():
    from otx.algorithms.common.adapters.mmcv.hooks.xpu_optimizer_hook import BFp16XPUOptimizerHook
    from otx.algorithms.common.adapters.torch.amp import XPUGradScaler

    hook = BFp16XPUOptimizerHook(grad_clip=None, coalesce=True, bucket_size_mb=-1, loss_scale=512.0, distributed=True)
    assert hook.coalesce is True  # Check coalesce is True
    assert hook.bucket_size_mb == -1  # Check bucket size is -1
    assert hook._scale_update_param is 512.0  # Check scale update param is 512.0
    assert hook.distributed is True  # Check distributed is True
    assert isinstance(hook.loss_scaler, XPUGradScaler)
