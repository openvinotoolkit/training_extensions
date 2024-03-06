import pytest
from otx.algorithms.common.utils.utils import is_xpu_available


def test_init():
    if not is_xpu_available():
        pytest.skip("XPU is not available")

    from otx.algorithms.common.adapters.mmcv.hooks.xpu_optimizer_hook import BFp16XPUOptimizerHook

    hook = BFp16XPUOptimizerHook(grad_clip=None, coalesce=True, bucket_size_mb=-1, loss_scale=512.0, distributed=True)
    assert hook._scale_update_param is None  # Check scale update param is None
    assert hook.coalesce is True  # Check coalesce is True
    assert hook.bucket_size_mb == -1  # Check bucket size is -1
    assert hook.loss_scale == 512.0  # Check loss scale is 512.0
    assert hook.distributed is True  # Check distributed is True
