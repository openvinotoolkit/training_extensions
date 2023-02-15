# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from functools import partial

from otx.algorithms.common.adapters.nncf.patches import (
    nncf_trace_wrapper,
    no_nncf_trace_wrapper,
)
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.algorithms.common.adapters.mmcv.nncf.test_helpers import create_model


@e2e_pytest_unit
def test_nncf_trace_context():
    model = create_model()
    bak = model.forward

    with model.nncf_trace_context({}, True):
        assert isinstance(model.forward, partial)
        assert model.forward.func == bak
    assert not isinstance(model.forward, partial)
    assert model.forward == bak

    with model.nncf_trace_context({}, False):
        assert isinstance(model.forward, partial)
        assert model.forward.func == model.forward_dummy
    assert not isinstance(model.forward, partial)
    assert model.forward == bak


@e2e_pytest_unit
def test_nncf_trace_wrapper():
    nncf_trace_wrapper("", lambda: None)


@e2e_pytest_unit
def test_no_nncf_trace_wrapper():
    no_nncf_trace_wrapper("", lambda: None)
