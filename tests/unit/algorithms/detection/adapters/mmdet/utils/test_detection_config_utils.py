"""Unit Test for otx.algorithms.detection.adapters.mmdet.utils.config_utils."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import math
import tempfile
from typing import List

import pytest
from mmcv.utils import Config, ConfigDict

from otx.algorithms.common.adapters.mmcv.utils import is_epoch_based_runner
from otx.algorithms.common.adapters.mmcv.utils.config_utils import MPAConfig
from otx.algorithms.detection.adapters.mmdet.utils.config_utils import (
    cluster_anchors,
    patch_adaptive_repeat_dataset,
    patch_config,
    patch_datasets,
    patch_evaluation,
    patch_model_config,
    prepare_for_training,
    set_hyperparams,
    should_cluster_anchors,
)
from otx.algorithms.detection.configs.base import DetectionConfig
from otx.api.entities.label import Domain
from otx.api.entities.model_template import TaskType
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.algorithms.detection.test_helpers import (
    DEFAULT_DET_MODEL_CONFIG_PATH,
    DEFAULT_ISEG_MODEL_CONFIG_PATH,
    generate_det_dataset,
    generate_labels,
)


@e2e_pytest_unit
@pytest.mark.parametrize("domain", [Domain.DETECTION, Domain.INSTANCE_SEGMENTATION])
def test_patch_config(domain):
    """Test patch_config function."""
    config = Config(
        dict(
            evaluation=dict(),
            checkpoint_config=dict(),
            data=dict(train=dict(labels=dict()), val=dict(labels=dict()), test=dict(labels=dict())),
        )
    )
    labels = generate_labels(1, domain)
    with tempfile.TemporaryDirectory() as work_dir:
        patch_config(config, work_dir, labels)
    assert "train_pipeline" not in config
    assert "test_pipeline" not in config
    assert "gpu_ids" in config
    assert "work_dir" in config
    assert "max_keep_ckpts" in config.checkpoint_config
    assert "interval" in config.checkpoint_config
    assert "labels" in config.data.train
    assert "labels" in config.data.val
    assert "labels" in config.data.test


@e2e_pytest_unit
@pytest.mark.parametrize(
    "cfg_path, domain",
    [
        (DEFAULT_DET_MODEL_CONFIG_PATH, Domain.DETECTION),
        (
            DEFAULT_ISEG_MODEL_CONFIG_PATH,
            Domain.INSTANCE_SEGMENTATION,
        ),
    ],
)
@pytest.mark.parametrize("num_classes", [1, 3, 10, 80])
def test_patch_model_config(num_classes, cfg_path, domain):
    """Test patch_model_config function."""
    config = MPAConfig.fromfile(cfg_path)
    labels = generate_labels(num_classes, domain)
    head_names = ("mask_head", "bbox_head", "segm_head")
    patch_model_config(config, labels)
    if "roi_head" in config.model:
        for head_name in head_names:
            if head_name in config.model.roi_head:
                if isinstance(config.model.roi_head[head_name], List):
                    for head in config.model.roi_head[head_name]:
                        assert head.num_classes == 1
                else:
                    assert config.model.roi_head[head_name].num_classes == num_classes
    else:
        for head_name in head_names:
            if head_name in config.model:
                assert config.model[head_name].num_classes == num_classes


@e2e_pytest_unit
def test_set_hyperparams():
    """Test set_hyperparams function."""
    config = Config(dict(data=dict(), optimizer=dict(), lr_config=dict(), runner=dict(type="EpochBasedRunner")))
    hyperparams = DetectionConfig()
    set_hyperparams(config, hyperparams)
    assert config.data.samples_per_gpu == int(hyperparams.learning_parameters.batch_size)
    assert config.data.workers_per_gpu == int(hyperparams.learning_parameters.num_workers)
    assert config.optimizer.lr == float(hyperparams.learning_parameters.learning_rate)
    assert config.lr_config.warmup_iters == int(hyperparams.learning_parameters.learning_rate_warmup_iters)
    total_iterations = int(hyperparams.learning_parameters.num_iters)
    if is_epoch_based_runner(config.runner):
        assert config.runner.max_epochs == total_iterations
    else:
        assert config.runner.max_iters == total_iterations


@e2e_pytest_unit
@pytest.mark.parametrize("num_samples", [10, 10000])
@pytest.mark.parametrize("decay", [-0.002, -0.2])
@pytest.mark.parametrize("factor", [30, 3000])
@pytest.mark.parametrize("dataset_type", ["MultiImageMixDataset", "RepeatDataset"])
def test_patch_adaptive_repeat_dataset(num_samples, decay, factor, dataset_type):
    """Test patch_adaptive_repeat function."""
    cur_epoch, cur_repeat = 2, 1
    config = Config(
        dict(
            runner=dict(type="EpochBasedRunner", max_epochs=cur_epoch),
            data=dict(train=dict(type=dataset_type, adaptive_repeat_times=True, times=cur_repeat)),
        )
    )
    if dataset_type == "MultiImageMixDataset":
        config.merge_from_dict(
            dict(
                data=dict(train=dict(dataset=dict(type="RepeatDataset", adaptive_repeat_times=True, times=cur_repeat)))
            )
        )
    patch_adaptive_repeat_dataset(config, num_samples=num_samples, decay=decay, factor=factor)
    adaptive_repeat = max(round(math.exp(decay * num_samples) * factor), 1)
    adaptive_epoch = math.ceil(cur_epoch / num_samples)
    if adaptive_epoch == 1:
        assert config.data.train.times == cur_repeat
        assert config.runner.max_epochs == cur_epoch
    else:
        assert config.data.train.times == adaptive_repeat
        assert config.runner.max_epochs == adaptive_epoch


@e2e_pytest_unit
def test_prepare_for_training():
    """Test prepare_for_training function."""
    with tempfile.TemporaryDirectory() as work_dir:
        config = Config(
            dict(
                data=dict(
                    train=dict(type="RepeatDataset", adaptive_repeat_times=True, otx_dataset=[]),
                    val=dict(),
                    test=dict(),
                ),
                work_dir=work_dir,
                runner=dict(type="EpochBasedRunner", max_epochs=1),
            )
        )
        data_config = ConfigDict(data=dict(train=dict(otx_dataset=[1, 2, 3]), val=dict(), test=dict()))
        prepare_for_training(config, data_config)
        assert "meta" in config.runner
        assert "checkpoints_round_0" in config.work_dir
        assert config.data.train.otx_dataset == [1, 2, 3]


@e2e_pytest_unit
def test_patch_datasets():
    """Test patch_datasets function."""
    config = Config(
        dict(
            data=dict(train=dict(), val=dict(), test=dict(), unlabeled=dict()),
        )
    )
    patch_datasets(config, type="FakeType")
    data_cfg = config.data
    assert "domain" in data_cfg.train
    assert "domain" in data_cfg.val
    assert "domain" in data_cfg.test
    assert "domain" in data_cfg.unlabeled
    assert "otx_dataset" in data_cfg.train
    assert "otx_dataset" in data_cfg.val
    assert "otx_dataset" in data_cfg.test
    assert "otx_dataset" in data_cfg.unlabeled
    assert "labels" in data_cfg.train
    assert "labels" in data_cfg.val
    assert "labels" in data_cfg.test
    assert "labels" in data_cfg.unlabeled
    assert "train_dataloader" in data_cfg
    assert "val_dataloader" in data_cfg
    assert "test_dataloader" in data_cfg


@e2e_pytest_unit
def test_patch_evaluation():
    """Test patch_evaluation function."""
    config = Config(dict(evaluation=dict()))
    patch_evaluation(config)
    assert "metric" in config.evaluation
    assert "save_best" in config.evaluation
    assert "early_stop_metric" in config


@e2e_pytest_unit
@pytest.mark.parametrize("reclustering_anchors", [True, False])
def test_should_cluster_anchors(reclustering_anchors):
    """Test should_cluster_anchors function."""
    config = Config(
        dict(
            model=dict(
                bbox_head=dict(
                    anchor_generator=dict(reclustering_anchors=reclustering_anchors),
                )
            )
        )
    )
    out = should_cluster_anchors(config)
    assert out == reclustering_anchors


@e2e_pytest_unit
@pytest.mark.parametrize("widths, heights", [([1, 10], [1, 10]), ([1, 3, 5, 7, 9], [1, 3, 5, 7, 9])])
def test_cluster_anchors(widths, heights):
    """Test cluster_anchors function."""
    recipe_config = Config(
        dict(
            model=dict(bbox_head=dict(anchor_generator=dict(widths=[widths], heights=[heights]))),
            data=dict(
                test=dict(
                    pipeline=[
                        dict(type="LoadImageFromFile"),
                        dict(
                            type="MultiScaleFlipAug",
                            img_scale=(10, 10),
                        ),
                    ]
                )
            ),
        )
    )
    dataset, _ = generate_det_dataset(task_type=TaskType.DETECTION)
    cluster_anchors(recipe_config, dataset)
