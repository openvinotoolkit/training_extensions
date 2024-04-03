# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np

from otx.algorithms.common.adapters.mmdeploy.utils.utils import numpy_2_list
from tests.test_suite.e2e_test_system import e2e_pytest_unit


@e2e_pytest_unit
def test_numpy2list():
    assert (0,) == numpy_2_list((0,))
    assert [0] == numpy_2_list([0])
    assert 0 == numpy_2_list(0)
    assert {0: 0} == numpy_2_list({0: 0})
    assert [0] == numpy_2_list(np.array([0]))
    assert 0 == numpy_2_list(np.array(0))
    assert {0: 0} == numpy_2_list({0: np.array(0)})
