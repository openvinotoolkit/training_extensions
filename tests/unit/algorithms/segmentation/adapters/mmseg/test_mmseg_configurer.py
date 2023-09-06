"""Test otx mmseg configurer."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import copy
import os

import pytest
import tempfile
from mmcv.utils import ConfigDict

from otx.algorithms.common.adapters.mmcv.utils.config_utils import MPAConfig
from otx.algorithms.segmentation.adapters.mmseg import configurer
from otx.algorithms.segmentation.adapters.mmseg.configurer import (
    SegmentationConfigurer,
    IncrSegmentationConfigurer,
    SemiSLSegmentationConfigurer,
)
from otx.algorithms.common.configs.configuration_enums import InputSizePreset
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.algorithms.segmentation.test_helpers import (
    DEFAULT_SEG_TEMPLATE_DIR,
)


class TestSegmentationConfigurer:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.configurer = SegmentationConfigurer("segmentation", True)
        self.model_cfg = MPAConfig.fromfile(os.path.join(DEFAULT_SEG_TEMPLATE_DIR, "model.py"))
        data_pipeline_cfg = MPAConfig.fromfile(os.path.join(DEFAULT_SEG_TEMPLATE_DIR, "data_pipeline.py"))
        self.model_cfg.merge_from_dict(data_pipeline_cfg)
        self.data_cfg = MPAConfig(
            {
                "data": {
                    "train": {"otx_dataset": [], "labels": []},
                    "val": {"otx_dataset": [], "labels": []},
                    "test": {"otx_dataset": [], "labels": []},
                }
            }
        )

    @e2e_pytest_unit
    def test_configure(self, mocker):
        mock_cfg_ckpt = mocker.patch.object(SegmentationConfigurer, "configure_ckpt")
        mock_cfg_data = mocker.patch.object(SegmentationConfigurer, "configure_data")
        mock_cfg_env = mocker.patch.object(SegmentationConfigurer, "configure_env")
        mock_cfg_data_pipeline = mocker.patch.object(SegmentationConfigurer, "configure_data_pipeline")
        mock_cfg_recipe = mocker.patch.object(SegmentationConfigurer, "configure_recipe")
        mock_cfg_model = mocker.patch.object(SegmentationConfigurer, "configure_model")
        mock_cfg_compat_cfg = mocker.patch.object(SegmentationConfigurer, "configure_compat_cfg")

        model_cfg = copy.deepcopy(self.model_cfg)
        data_cfg = copy.deepcopy(self.data_cfg)
        returned_value = self.configurer.configure(model_cfg, "", data_cfg)
        mock_cfg_ckpt.assert_called_once_with(model_cfg, "")
        mock_cfg_data.assert_called_once_with(model_cfg, data_cfg)
        mock_cfg_env.assert_called_once_with(model_cfg)
        mock_cfg_data_pipeline.assert_called_once_with(model_cfg, InputSizePreset.DEFAULT, "")
        mock_cfg_recipe.assert_called_once_with(model_cfg)
        mock_cfg_model.assert_called_once_with(model_cfg, None, None, None)
        mock_cfg_compat_cfg.assert_called_once_with(model_cfg)
        assert returned_value == model_cfg

    @e2e_pytest_unit
    def test_configure_ckpt(self, mocker):
        model_cfg = copy.deepcopy(self.model_cfg)
        model_cfg.resume = True

        mocker.patch(
            "otx.algorithms.segmentation.adapters.mmseg.configurer.CheckpointLoader.load_checkpoint",
            return_value={"model": None},
        )
        with tempfile.TemporaryDirectory() as tempdir:
            self.configurer.configure_ckpt(model_cfg, os.path.join(tempdir, "dummy.pth"))

    @e2e_pytest_unit
    def test_configure_data(self, mocker):
        data_cfg = copy.deepcopy(self.data_cfg)
        self.configurer.configure_data(self.model_cfg, data_cfg)
        assert self.model_cfg.data
        assert self.model_cfg.data.train
        assert self.model_cfg.data.val

    @e2e_pytest_unit
    def test_configure_env(self):
        self.configurer.configure_env(self.model_cfg)

    @e2e_pytest_unit
    def test_configure_device(self, mocker):
        mocker.patch(
            "torch.distributed.is_initialized",
            return_value=True,
        )
        mocker.patch(
            "torch.distributed.get_world_size",
            return_value=2,
        )
        world_size = 2
        mocker.patch("os.environ", return_value={"LOCAL_RANK": 2})
        config = copy.deepcopy(self.model_cfg)
        origin_lr = config.optimizer.lr
        self.configurer.configure_device(config)
        assert config.distributed is True
        assert config.optimizer.lr == pytest.approx(origin_lr * world_size)

        mocker.patch(
            "torch.distributed.is_initialized",
            return_value=False,
        )
        mocker.patch(
            "torch.cuda.is_available",
            return_value=False,
        )
        config = copy.deepcopy(self.model_cfg)
        self.configurer.configure_device(config)
        assert config.distributed is False
        assert config.device == "cpu"

        mocker.patch(
            "torch.distributed.is_initialized",
            return_value=False,
        )
        mocker.patch(
            "torch.cuda.is_available",
            return_value=True,
        )
        config = copy.deepcopy(self.model_cfg)
        self.configurer.configure_device(config)
        assert config.distributed is False
        assert config.device == "cuda"

    @e2e_pytest_unit
    def test_configure_samples_per_gpu(self):
        model_cfg = copy.deepcopy(self.model_cfg)
        model_cfg.data.train_dataloader = ConfigDict({"samples_per_gpu": 2})
        model_cfg.data.train.otx_dataset = range(1)
        self.configurer.configure_samples_per_gpu(model_cfg)
        assert model_cfg.data.train_dataloader == {"samples_per_gpu": 1, "drop_last": True}

    @e2e_pytest_unit
    @pytest.mark.parametrize("input_size", [None, (256, 256)])
    def test_configure_input_size(self, mocker, input_size):
        # prepare
        mock_cfg = mocker.MagicMock()
        mocker.patch.object(configurer, "get_configured_input_size", return_value=input_size)
        mock_input_manager = mocker.MagicMock()
        mock_input_manager_cls = mocker.patch.object(configurer, "InputSizeManager")
        mock_input_manager_cls.return_value = mock_input_manager
        base_input_size = {
            "train": 512,
            "val": 544,
            "test": 544,
            "unlabeled": 512,
        }

        # excute
        self.configurer.configure_input_size(mock_cfg, InputSizePreset.DEFAULT, self.data_cfg)

        # check
        if input_size is not None:
            mock_input_manager_cls.assert_called_once_with(mock_cfg.data, base_input_size)
            mock_input_manager.set_input_size.assert_called_once_with(input_size)
        else:
            mock_input_manager_cls.assert_not_called()

    @e2e_pytest_unit
    def test_configure_fp16(self):
        model_cfg = copy.deepcopy(self.model_cfg)
        model_cfg.fp16 = {}
        self.configurer.configure_fp16(model_cfg)
        assert model_cfg.optimizer_config.type == "Fp16OptimizerHook"

        model_cfg.fp16 = {}
        model_cfg.optimizer_config.type = "SAMOptimizerHook"
        self.configurer.configure_fp16(model_cfg)
        assert model_cfg.optimizer_config.type == "Fp16SAMOptimizerHook"

        model_cfg.fp16 = {}
        model_cfg.optimizer_config.type = "DummyOptimizerHook"
        self.configurer.configure_fp16(model_cfg)
        assert model_cfg.optimizer_config.type == "DummyOptimizerHook"

    @e2e_pytest_unit
    def test_configure_model(self):
        ir_options = {"ir_model_path": {"ir_weight_path": "", "ir_weight_init": ""}}
        self.configurer.configure_model(self.model_cfg, [], [], ir_options)
        assert self.model_cfg.model_task

    @e2e_pytest_unit
    def test_configure_task(self, mocker):
        model_cfg = copy.deepcopy(self.model_cfg)
        mock_cfg_classes = mocker.patch.object(SegmentationConfigurer, "configure_classes")
        mock_cfg_ignore = mocker.patch.object(SegmentationConfigurer, "configure_decode_head")
        self.configurer.configure_task(model_cfg)

        mock_cfg_classes.assert_called_once()
        mock_cfg_ignore.assert_called_once()

    @e2e_pytest_unit
    def test_configure_decode_head(self):
        model_cfg = copy.deepcopy(self.model_cfg)
        self.configurer.configure_decode_head(model_cfg)
        if "decode_head" in model_cfg.model:
            assert model_cfg.model.decode_head.loss_decode.type == "CrossEntropyLossWithIgnore"

    @e2e_pytest_unit
    def test_configure_classes_replace(self, mocker):
        model_cfg = copy.deepcopy(self.model_cfg)
        self.configurer.task_adapt_op = "REPLACE"
        mocker.patch.object(SegmentationConfigurer, "get_data_classes", return_value=["foo", "bar"])
        self.configurer.configure_classes(model_cfg)
        assert "background" in self.configurer.model_classes
        assert self.configurer.model_classes == ["background", "foo", "bar"]

    @e2e_pytest_unit
    def test_configure_classes_merge(self, mocker):
        model_cfg = copy.deepcopy(self.model_cfg)
        self.configurer.task_adapt_op = "MERGE"
        mocker.patch.object(SegmentationConfigurer, "get_model_classes", return_value=["foo", "bar"])
        mocker.patch.object(SegmentationConfigurer, "get_data_classes", return_value=["foo", "baz"])
        self.configurer.configure_classes(model_cfg)
        assert "background" in self.configurer.model_classes
        assert self.configurer.model_classes == ["background", "foo", "bar", "baz"]

    @e2e_pytest_unit
    def test_configure_compat_cfg(self):
        model_cfg = copy.deepcopy(self.model_cfg)
        model_cfg.data.train_dataloader = {}
        model_cfg.data.val_dataloader = {}
        model_cfg.data.test_dataloader = {}
        self.configurer.configure_compat_cfg(model_cfg)


class TestIncrSegmentationConfigurer:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.configurer = IncrSegmentationConfigurer("segmentation", True)
        self.model_cfg = MPAConfig.fromfile(os.path.join(DEFAULT_SEG_TEMPLATE_DIR, "model.py"))
        data_pipeline_cfg = MPAConfig.fromfile(os.path.join(DEFAULT_SEG_TEMPLATE_DIR, "data_pipeline.py"))
        self.model_cfg.merge_from_dict(data_pipeline_cfg)
        self.data_cfg = MPAConfig(
            {
                "data": {
                    "train": {"otx_dataset": [], "labels": []},
                    "val": {"otx_dataset": [], "labels": []},
                    "test": {"otx_dataset": [], "labels": []},
                }
            }
        )

    @e2e_pytest_unit
    def test_configure_task(self, mocker):
        mocker.patch.object(SegmentationConfigurer, "configure_task")
        self.configurer.task_adapt_type = "default_task_adapt"
        self.configurer.configure_task(self.model_cfg)
        assert self.model_cfg.custom_hooks[3].type == "TaskAdaptHook"
        assert self.model_cfg.custom_hooks[3].sampler_flag is False


class TestSemiSLSegmentationConfigurer:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.configurer = SemiSLSegmentationConfigurer("segmentation", True)
        self.cfg = MPAConfig.fromfile(os.path.join(DEFAULT_SEG_TEMPLATE_DIR, "semisl", "model.py"))
        data_pipeline_cfg = MPAConfig.fromfile(os.path.join(DEFAULT_SEG_TEMPLATE_DIR, "semisl", "data_pipeline.py"))
        self.cfg.merge_from_dict(data_pipeline_cfg)

    @e2e_pytest_unit
    def test_configure_data(self, mocker):
        mocker.patch("otx.algorithms.common.adapters.mmcv.semisl_mixin.build_dataset", return_value=True)
        mocker.patch("otx.algorithms.common.adapters.mmcv.semisl_mixin.build_dataloader", return_value=True)

        data_cfg = MPAConfig(
            {
                "data": {
                    "train": {"otx_dataset": [], "labels": []},
                    "val": {"otx_dataset": [], "labels": []},
                    "test": {"otx_dataset": [], "labels": []},
                    "unlabeled": {"otx_dataset": [0, 1, 2, 3], "labels": []},
                }
            }
        )
        self.cfg.model_task = "classification"
        self.cfg.distributed = False
        self.configurer.configure_data(self.cfg, data_cfg)
        assert self.cfg.custom_hooks[-1]["type"] == "ComposedDataLoadersHook"
