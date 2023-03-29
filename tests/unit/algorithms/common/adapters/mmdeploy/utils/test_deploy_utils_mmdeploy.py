# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import importlib

from mmcv.utils import Config

from otx.algorithms.common.adapters.mmdeploy.utils.mmdeploy import (
    is_mmdeploy_enabled,
    mmdeploy_init_model_helper,
)
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.algorithms.common.adapters.mmdeploy.test_helpers import (
    create_config,
    create_model,
)


@e2e_pytest_unit
def test_is_mmdeploy_enabled():
    assert (importlib.util.find_spec("mmdeploy") is not None) == is_mmdeploy_enabled()


@e2e_pytest_unit
def test_mmdeploy_init_model_helper():
    from otx.algorithms.classification.adapters.mmcls.utils.builder import (
        build_classifier,
    )

    config = Config(
        {
            "model_cfg": create_config(),
            "device": "cpu",
        }
    )

    if importlib.util.find_spec("mmcls"):
        create_model("mmcls")
        model = mmdeploy_init_model_helper(config, model_builder=build_classifier)
        for i in model.parameters():
            assert not i.requires_grad
