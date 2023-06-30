"""Unit Test for otx.algorithms.detection.adapters.mmdet.utils.config_utils."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import pytest
from mmcv.utils import Config

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
