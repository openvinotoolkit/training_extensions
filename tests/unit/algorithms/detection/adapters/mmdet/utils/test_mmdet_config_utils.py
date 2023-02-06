"""Unit Test for otx.algorithms.action.adapters.detection.utils.config_utils."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
from typing import List

import pytest
from mmcv.utils import Config, ConfigDict

from otx.algorithms.common.adapters.mmcv.utils import is_epoch_based_runner
from otx.algorithms.detection.adapters.mmdet.utils.config_utils import (
    patch_adaptive_repeat_dataset,
    patch_config,
    patch_model_config,
    prepare_for_training,
    set_hyperparams,
)
from otx.algorithms.detection.configs.base import DetectionConfig
from otx.api.entities.label import Domain
from otx.mpa.utils.config_utils import MPAConfig
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.algorithms.detection.test_helpers import generate_labels


class TestOTXDetConfigUtils:
    """Test OTXDetConfigUtils methods."""

    @e2e_pytest_unit
    @pytest.mark.parametrize("domain", [Domain.DETECTION, Domain.INSTANCE_SEGMENTATION])
    def test_patch_config(self, domain):
        """Test build_detector method."""
        config = Config()
        options = dict(
            evaluation=dict(),
            checkpoint_config=dict(),
            data=dict(train=dict(labels=dict()), val=dict(labels=dict()), test=dict(labels=dict())),
        )
        config.merge_from_dict(options)
        work_dir = "./some_work_dir"
        labels = generate_labels(3, domain)
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
            ("./otx/algorithms/detection/configs/detection/mobilenetv2_atss/model.py", Domain.DETECTION),
            (
                "./otx/algorithms/detection/configs/instance_segmentation/efficientnetb2b_maskrcnn/model.py",
                Domain.INSTANCE_SEGMENTATION,
            ),
        ],
    )
    def test_patch_model_config(self, cfg_path, domain):
        """Test build_detector method."""
        num_classes = 3
        config = MPAConfig.fromfile(cfg_path)
        labels = generate_labels(num_classes, domain)
        head_names = ("mask_head", "bbox_head", "segm_head")
        patch_model_config(config, labels)
        if "roi_head" in config.model:
            for head_name in head_names:
                if head_name in config.model.roi_head:
                    if isinstance(config.model.roi_head[head_name], List):
                        for head in config.model.roi_head[head_name]:
                            assert head.num_classes == 3
                    else:
                        assert config.model.roi_head[head_name].num_classes == num_classes
        else:
            for head_name in head_names:
                if head_name in config.model:
                    assert config.model[head_name].num_classes == num_classes

    @e2e_pytest_unit
    def test_set_hyperparams(self):
        """Test build_detector method."""
        config = Config()
        options = dict(data=dict(), optimizer=dict(), lr_config=dict(), runner=dict(type="EpochBasedRunner"))
        config.merge_from_dict(options)
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

    # TODO[Jihwan] - add various test codes, assert
    @e2e_pytest_unit
    @pytest.mark.parametrize("dataset_type", ["MultiImageMixDataset", "RepeatDataset"])
    @pytest.mark.parametrize("num_samples", [1, 3000])
    def test_patch_adaptive_repeat_dataset(self, num_samples, dataset_type):
        """Test build_detector method."""
        config = Config()
        options = dict(
            runner=dict(type="EpochBasedRunner", max_epochs=3),
            data=dict(train=dict(type=dataset_type, adaptive_repeat_times=True)),
        )
        if dataset_type == "MultiImageMixDataset":
            options["data"]["train"].update(dataset=dict(type="RepeatDataset", adaptive_repeat_times=True))
        config.merge_from_dict(options)
        patch_adaptive_repeat_dataset(config, num_samples=num_samples)

    # TODO[Jihwan] - add various test codes, assert, task랑 stage하면 자동으로 채워질듯?
    @e2e_pytest_unit
    def test_prepare_for_training(self):
        """Test build_detector method."""
        config = Config()
        config_options = dict(
            data=dict(
                train=dict(type="RepeatDataset", adaptive_repeat_times=True, otx_dataset=[]), val=dict(), test=dict()
            ),
            work_dir="./some_work_dir",
            runner=dict(type="EpochBasedRunner", max_epochs=1),
        )
        config.merge_from_dict(config_options)

        data_config = ConfigDict(data=dict(train=dict(otx_dataset=[1, 2, 3]), val=dict(), test=dict()))
        # data_config.merge_from_dict(data_config_options)
        prepare_for_training(config, data_config)
