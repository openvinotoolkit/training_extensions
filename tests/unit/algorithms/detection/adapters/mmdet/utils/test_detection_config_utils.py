"""Unit Test for otx.algorithms.detection.adapters.mmdet.utils.config_utils."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import pytest
from mmcv.utils import Config

from pathlib import Path

from otx.algorithms.detection.adapters.mmdet.utils.config_utils import (
    cluster_anchors,
    patch_datasets,
    patch_evaluation,
    should_cluster_anchors,
)
from otx.api.entities.model_template import TaskType
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.algorithms.detection.test_helpers import (
    generate_det_dataset,
)

from otx.algorithms.common.adapters.mmcv.utils.config_utils import MPAConfig

from tests.unit.algorithms.detection.test_helpers import (
    DEFAULT_DET_MODEL_CONFIG_PATH,
    DEFAULT_ISEG_MODEL_CONFIG_PATH,
)
from otx.api.configuration.helper import create
from otx.api.entities.model_template import parse_model_template

from otx.algorithms.detection.adapters.mmdet.utils import patch_samples_per_gpu


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
def test_patch_datasets_disable_memcache_for_test_subset():
    """Test patch_datasets function to check memcache disabled for test set."""
    config = Config(
        dict(
            data=dict(
                train=dict(pipeline=[dict(type="LoadImageFromFile")]),
                val=dict(pipeline=[dict(type="LoadImageFromFile")]),
                test=dict(pipeline=[dict(type="LoadImageFromFile")]),
                unlabeled=dict(pipeline=[dict(type="LoadImageFromFile")]),
            )
        )
    )
    patch_datasets(config, type="FakeType")
    assert config.data.train.pipeline[0].type == "LoadImageFromOTXDataset"
    assert config.data.train.pipeline[0].enable_memcache == True
    assert config.data.val.pipeline[0].type == "LoadImageFromOTXDataset"
    assert config.data.val.pipeline[0].enable_memcache == True
    assert config.data.test.pipeline[0].type == "LoadImageFromOTXDataset"
    assert getattr(config.data.test.pipeline[0], "enable_memcache", False) == False
    # Note: cannot set enable_memcache attr due to mmdeploy error
    assert config.data.unlabeled.pipeline[0].type == "LoadImageFromOTXDataset"
    assert config.data.unlabeled.pipeline[0].enable_memcache == True


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


@e2e_pytest_unit
@pytest.mark.parametrize("model_cfg", [DEFAULT_DET_MODEL_CONFIG_PATH, DEFAULT_ISEG_MODEL_CONFIG_PATH])
def test_patch_samples_per_gpu(model_cfg):
    """Test that patch_samples_per_gpu function works correctly."""
    cfg = MPAConfig.fromfile(model_cfg)
    model_template = parse_model_template(Path(model_cfg).parent / "template.yaml")
    hyper_parameters = create(model_template.hyper_parameters.data)

    patch_samples_per_gpu(cfg, hyper_parameters)
    params = hyper_parameters.learning_parameters
    assert cfg.data.val_dataloader.samples_per_gpu == params.inference_batch_size
    assert cfg.data.test_dataloader.samples_per_gpu == params.inference_batch_size
