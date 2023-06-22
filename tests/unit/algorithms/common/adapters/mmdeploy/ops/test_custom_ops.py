"""Test for otx.algorithms.common.adapters.mmdeploy.ops.custom_ops."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch
from mmcv.utils import Config
from mmdeploy.core import SYMBOLIC_REWRITER

from otx.algorithms.common.adapters.mmdeploy.ops.custom_ops import squeeze__default
from tests.test_suite.e2e_test_system import e2e_pytest_unit


@e2e_pytest_unit
def test_symbolic_registery():
    assert len(SYMBOLIC_REWRITER._registry._rewrite_records["squeeze"]) == 1


class MockOps:
    def op(self, *args, **kwargs):
        return (args, kwargs)


@e2e_pytest_unit
def test_squeeze(mocker):
    """Test squeeze__default function."""

    class MockClass:
        class _size:
            def sizes(self):
                return [1, 1, 1]

        size = _size()

        def type(self):
            return self.size

    # Patching for squeeze op
    mock_ctx = Config({"cfg": Config({"opset_version": 11})})
    mock_g = MockOps()
    mock_self = MockClass()
    mocker.patch("otx.algorithms.common.adapters.mmdeploy.ops.custom_ops.get_ir_config", return_value=mock_ctx.cfg)
    op = squeeze__default(mock_ctx, mock_g, mock_self)
    assert op[0][0] == "Squeeze"
    assert op[1]["axes_i"] == [0, 1, 2]

    mock_ctx = Config({"cfg": Config({"opset_version": 13})})
    mock_g = MockOps()
    mock_self = MockClass()
    mocker.patch("otx.algorithms.common.adapters.mmdeploy.ops.custom_ops.get_ir_config", return_value=mock_ctx.cfg)
    op = squeeze__default(mock_ctx, mock_g, mock_self)
    assert op[0][0] == "Squeeze"
    assert op[0][2][0][0] == "Constant"
    assert torch.all(op[0][2][1]["value_t"] == torch.Tensor([0, 1, 2]))
