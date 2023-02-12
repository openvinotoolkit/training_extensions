# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from mmcls.models.classifiers.base import BaseClassifier

import otx.algorithms.classification.adapters.mmcls.nncf.patches  # noqa: F401
from tests.test_suite.e2e_test_system import e2e_pytest_unit


@e2e_pytest_unit
def test_patches():
    assert getattr(BaseClassifier, "nncf_trace_context", None) is not None
