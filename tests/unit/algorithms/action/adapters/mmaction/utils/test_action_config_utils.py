"""Unit Test for otx.algorithms.action.adapters.mmaction.utils.config_utils."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import tempfile
from collections import defaultdict

import pytest
from mmcv.utils import Config

from otx.algorithms.action.adapters.mmaction.utils import (
    patch_config,
    prepare_for_training,
    set_data_classes,
)
from otx.algorithms.common.adapters.mmcv.utils import get_data_cfg
from otx.api.entities.annotation import NullAnnotationSceneEntity
from otx.api.entities.dataset_item import DatasetItemEntity
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.image import Image
from otx.api.entities.label import Domain, LabelEntity
from otx.api.entities.model_template import TaskType
from otx.api.usecases.reporting.time_monitor_callback import TimeMonitorCallback
from tests.test_suite.e2e_test_system import e2e_pytest_unit

CLS_CONFIG_NAME = "otx/algorithms/action/configs/classification/x3d/model.py"
DET_CONFIG_NAME = "otx/algorithms/action/configs/detection/x3d_fast_rcnn/model.py"
CLS_CONFIG = Config.fromfile(CLS_CONFIG_NAME)
DET_CONFIG = Config.fromfile(DET_CONFIG_NAME)


@e2e_pytest_unit
def test_patch_config() -> None:
    """Test patch_config function.

    <Steps>
        1. Check error when gives wrong task type
        2. Check work_dir
        3. Check merged data pipeline
        4. Check dataset type
    """

    cls_datapipeline_path = "otx/algorithms/action/configs/classification/x3d/data_pipeline.py"
    with tempfile.TemporaryDirectory() as work_dir:
        with pytest.raises(NotImplementedError):
            patch_config(CLS_CONFIG, cls_datapipeline_path, work_dir, TaskType.CLASSIFICATION)

        patch_config(CLS_CONFIG, cls_datapipeline_path, work_dir, TaskType.ACTION_CLASSIFICATION)
        assert CLS_CONFIG.work_dir == work_dir
        assert CLS_CONFIG.get("train_pipeline", None)
        for subset in ("train", "val", "test", "unlabeled"):
            cfg = CLS_CONFIG.data.get(subset, None)
            if not cfg:
                continue
            assert cfg.type == "OTXActionClsDataset"

        det_datapipeline_path = "otx/algorithms/action/configs/detection/x3d_fast_rcnn/data_pipeline.py"
        patch_config(DET_CONFIG, det_datapipeline_path, work_dir, TaskType.ACTION_DETECTION)
        assert DET_CONFIG.work_dir == work_dir
        assert DET_CONFIG.get("train_pipeline", None)
        for subset in ("train", "val", "test", "unlabeled"):
            cfg = DET_CONFIG.data.get(subset, None)
            if not cfg:
                continue
            assert cfg.type == "OTXActionDetDataset"


@e2e_pytest_unit
def test_set_data_classes() -> None:
    """Test set_data_classes funciton.

    <Step>
        1. Create sample labels
        2. Check classification label length is appropriate
    """

    labels = [
        LabelEntity(name="0", domain=Domain.ACTION_CLASSIFICATION),
        LabelEntity(name="1", domain=Domain.ACTION_CLASSIFICATION),
        LabelEntity(name="2", domain=Domain.ACTION_CLASSIFICATION),
    ]
    set_data_classes(CLS_CONFIG, labels, TaskType.ACTION_CLASSIFICATION)
    assert CLS_CONFIG.model["cls_head"].num_classes == len(labels)

    labels = [
        LabelEntity(name="0", domain=Domain.ACTION_DETECTION),
        LabelEntity(name="1", domain=Domain.ACTION_DETECTION),
        LabelEntity(name="2", domain=Domain.ACTION_DETECTION),
    ]
    set_data_classes(DET_CONFIG, labels, TaskType.ACTION_DETECTION)
    assert DET_CONFIG.model["roi_head"]["bbox_head"].num_classes == len(labels) + 1
    assert DET_CONFIG.model["roi_head"]["bbox_head"]["topk"] == len(labels) - 1


@e2e_pytest_unit
def test_prepare_for_training() -> None:
    """Test prepare_for_training function.

    <Step>
        1. Create sample DatasetEntity, TimeMonitorCallback, Learngin_cureves
        2. Check config.data.train
        3. Check config.data.val
        4. Check custom_hooks
        5. Check log_config.hooks
    """

    item = DatasetItemEntity(media=Image(file_path="iamge.jpg"), annotation_scene=NullAnnotationSceneEntity())
    dataset = DatasetEntity(items=[item])
    time_monitor = TimeMonitorCallback()
    learning_curves = defaultdict()

    CLS_CONFIG.runner = {}
    CLS_CONFIG.custom_hooks = []
    prepare_for_training(CLS_CONFIG, dataset, dataset, time_monitor, learning_curves)

    assert get_data_cfg(CLS_CONFIG, "train").otx_dataset == dataset
    assert get_data_cfg(CLS_CONFIG, "val").otx_dataset == dataset
    assert (
        CLS_CONFIG.custom_hooks[-1]["type"] == "OTXProgressHook"
        and CLS_CONFIG.custom_hooks[-1]["time_monitor"] == time_monitor
    )
    assert (
        CLS_CONFIG.log_config.hooks[-1]["type"] == "OTXLoggerHook"
        and CLS_CONFIG.log_config.hooks[-1]["curves"] == learning_curves
    )
