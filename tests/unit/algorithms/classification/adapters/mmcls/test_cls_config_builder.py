# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import tempfile

import pytest
from mmcv.utils import Config

from otx.algorithms.classification.adapters.mmcls.utils import (
    patch_config,
    patch_evaluation,
)
from otx.algorithms.common.adapters.mmcv.utils import get_dataset_configs
from otx.api.entities.id import ID
from otx.api.entities.label import Domain, LabelEntity
from tests.test_suite.e2e_test_system import e2e_pytest_unit


@pytest.fixture
def otx_default_cls_config():
    config_name = "otx/algorithms/classification/configs/efficientnet_b0_cls_incr/model.py"
    conf = Config.fromfile(config_name)
    conf.checkpoint_config = {}
    conf.evaluation = {"metric": "None"}
    conf.data = {}
    return conf


@pytest.fixture
def otx_default_labels():
    return [
        LabelEntity(name=name, domain=Domain.CLASSIFICATION, is_empty=False, id=ID(i))
        for i, name in enumerate(["a", "b"])
    ]


@e2e_pytest_unit
def test_patch_config(otx_default_cls_config, otx_default_labels) -> None:
    """Test patch_config function.

    <Steps>
        1. Check work_dir
        2. Check removed high level pipelines
        3. Check checkpoint config update
        4. Check dataset labels
    """

    with tempfile.TemporaryDirectory() as work_dir:
        patch_config(otx_default_cls_config, work_dir, otx_default_labels)
        assert otx_default_cls_config.work_dir == work_dir

        assert otx_default_cls_config.get("train_pipeline", None) is None
        assert otx_default_cls_config.get("test_pipeline", None) is None
        assert otx_default_cls_config.get("train_pipeline_strong", None) is None

        assert otx_default_cls_config.checkpoint_config.max_keep_ckpts > 0
        assert otx_default_cls_config.checkpoint_config.interval > 0

        for subset in ("train", "val", "test"):
            for cfg in get_dataset_configs(otx_default_cls_config, subset):
                assert len(cfg.labels) == len(otx_default_labels)


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
