# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import pytest

from otx.algorithms.classification.adapters.mmcls.utils import patch_evaluation
from otx.algorithms.common.adapters.mmcv.utils.config_utils import OTXConfig
from tests.test_suite.e2e_test_system import e2e_pytest_unit


@pytest.fixture
def otx_default_cls_config():
    config_name = "src/otx/algorithms/classification/configs/efficientnet_b0_cls_incr/model.py"
    conf = OTXConfig.fromfile(config_name)
    conf.checkpoint_config = {}
    conf.evaluation = {"metric": "None"}
    conf.data = {}
    return conf


@e2e_pytest_unit
def test_patch_evaluation(otx_default_cls_config) -> None:
    """Test patch_evaluation function.

    <Steps>
        1. Check eval metrics not empty
        2. Check the early stop metric is in eval metrics
        3. Check an exception is thrown for a frong task type
    """
    tasks = ["multilabel", "hierarchical", "normal"]
    for task in tasks:
        patch_evaluation(otx_default_cls_config, task)
        eval_metrics = otx_default_cls_config.evaluation.metric
        assert len(eval_metrics) > 0
        assert otx_default_cls_config.early_stop_metric in eval_metrics

    with pytest.raises(NotImplementedError):
        patch_evaluation(otx_default_cls_config, "None")
