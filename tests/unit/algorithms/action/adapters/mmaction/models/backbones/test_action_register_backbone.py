"""Unit test for otx.algorithms.action.adapters.mmaction.models.backbones."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
from mmaction.models import BACKBONES as MMACTION_BACKBONES
from mmdet.models import BACKBONES as MMDET_BACKBONES

from tests.test_suite.e2e_test_system import e2e_pytest_unit


@e2e_pytest_unit
def test_register_action_backbones() -> None:
    """Test register_action_backbones function.

    Since this function is called while initialization, X3D should be in mmdet backbone registry
    """

    assert "X3D" in MMDET_BACKBONES
    assert "X3D" in MMACTION_BACKBONES
    assert "OTXMoViNet" in MMACTION_BACKBONES
