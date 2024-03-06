"""Test for otx.algorithms.common.adapters.mmcv.configurer"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import pytest
from mmcv.utils import Config
from otx.algorithms.common.adapters.mmcv import configurer
from otx.algorithms.common.adapters.mmcv.utils.config_utils import InputSizeManager
from tests.test_suite.e2e_test_system import e2e_pytest_unit


@e2e_pytest_unit
class TestBaseConfigurer:
    def test_get_input_size_to_fit_dataset(self, mocker):
        cfg = Config({"data": {"train": {"otx_dataset": None}}})
        input_size_manager = InputSizeManager(cfg)
        input_size = configurer.BaseConfigurer.adapt_input_size_to_dataset(cfg, input_size_manager)
        assert input_size is None

        cfg = Config({"data": {"train": {"otx_dataset": True}}})
        input_size_manager = InputSizeManager(cfg, base_input_size=512)
        mock_stat = mocker.patch.object(configurer, "compute_robust_dataset_statistics")

        mock_stat.return_value = {}
        input_size = configurer.BaseConfigurer.adapt_input_size_to_dataset(cfg, input_size_manager)
        assert input_size is None

        mock_stat.return_value = dict(
            image=dict(robust_max=150),
        )
        input_size = configurer.BaseConfigurer.adapt_input_size_to_dataset(cfg, input_size_manager)
        assert input_size == (128, 128)

        mock_stat.return_value = dict(
            image=dict(robust_max=150),
            annotation=dict(),
        )
        input_size = configurer.BaseConfigurer.adapt_input_size_to_dataset(
            cfg, input_size_manager, use_annotations=True
        )
        assert input_size == (128, 128)

        mock_stat.return_value = dict(
            image=dict(robust_max=256),
            annotation=dict(size_of_shape=dict(robust_min=64)),
        )
        input_size = configurer.BaseConfigurer.adapt_input_size_to_dataset(
            cfg, input_size_manager, use_annotations=True
        )
        assert input_size == (256, 256)

        mock_stat.return_value = dict(
            image=dict(robust_max=1024),
            annotation=dict(size_of_shape=dict(robust_min=64)),
        )
        input_size = configurer.BaseConfigurer.adapt_input_size_to_dataset(
            cfg, input_size_manager, use_annotations=True
        )
        assert input_size == (512, 512)

        mock_stat.return_value = dict(
            image=dict(robust_max=2045),
            annotation=dict(size_of_shape=dict(robust_min=64)),
        )
        input_size = configurer.BaseConfigurer.adapt_input_size_to_dataset(
            cfg, input_size_manager, use_annotations=True
        )
        assert input_size == (512, 512)
