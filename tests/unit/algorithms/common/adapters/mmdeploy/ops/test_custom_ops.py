"""Test for otx.algorithms.common.adapters.mmdeploy.ops.custom_ops."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from mmdeploy.core import SYMBOLIC_REWRITER

from otx.algorithms.common.adapters.mmdeploy.ops.custom_ops import (
    squeeze__default,
    grid_sampler__default,
)
from tests.test_suite.e2e_test_system import e2e_pytest_unit


@e2e_pytest_unit
def test_symbolic_registery():
    assert len(SYMBOLIC_REWRITER._registry._rewrite_records["squeeze"]) == 1
    assert len(SYMBOLIC_REWRITER._registry._rewrite_records["grid_sampler"]) == 1
