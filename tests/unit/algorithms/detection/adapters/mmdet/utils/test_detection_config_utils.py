"""Unit Test for otx.algorithms.detection.adapters.mmdet.utils.config_utils."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import pytest
from mmcv.utils import Config

from pathlib import Path

from otx.algorithms.detection.adapters.mmdet.utils.config_utils import (
    cluster_anchors,
    should_cluster_anchors,
)
from otx.api.entities.model_template import TaskType
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.algorithms.detection.test_helpers import (
    generate_det_dataset,
)

from otx.algorithms.common.adapters.mmcv.utils.config_utils import OTXConfig, patch_from_hyperparams

from tests.unit.algorithms.detection.test_helpers import (
    DEFAULT_DET_MODEL_CONFIG_PATH,
    DEFAULT_ISEG_MODEL_CONFIG_PATH,
)
from otx.api.configuration.helper import create
from otx.api.entities.model_template import parse_model_template


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
    """Test samples per gpu function works correctly."""
    cfg = OTXConfig.fromfile(model_cfg)
    model_template = parse_model_template(Path(model_cfg).parent / "template.yaml")
    hyper_parameters = create(model_template.hyper_parameters.data)

    patch_from_hyperparams(cfg, hyper_parameters)
    params = hyper_parameters.learning_parameters
    assert cfg.data.val_dataloader.samples_per_gpu == params.inference_batch_size
    assert cfg.data.test_dataloader.samples_per_gpu == params.inference_batch_size
