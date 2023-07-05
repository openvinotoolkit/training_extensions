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
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.algorithms.segmentation.test_helpers import (
    DEFAULT_SEG_TEMPLATE_DIR,
)


class TestSegmentationConfigurer:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.configurer = SegmentationConfigurer()
        self.model_cfg = MPAConfig.fromfile(os.path.join(DEFAULT_SEG_TEMPLATE_DIR, "model.py"))
        self.data_cfg = MPAConfig.fromfile(os.path.join(DEFAULT_SEG_TEMPLATE_DIR, "data_pipeline.py"))

    @e2e_pytest_unit
    def test_configure(self, mocker):
        mock_cfg_base = mocker.patch.object(SegmentationConfigurer, "configure_base")
        mock_cfg_device = mocker.patch.object(SegmentationConfigurer, "configure_device")
        mock_cfg_ckpt = mocker.patch.object(SegmentationConfigurer, "configure_ckpt")
        mock_cfg_model = mocker.patch.object(SegmentationConfigurer, "configure_model")
        mock_cfg_task = mocker.patch.object(SegmentationConfigurer, "configure_task")
        mock_cfg_hook = mocker.patch.object(SegmentationConfigurer, "configure_hook")
        mock_cfg_gpu = mocker.patch.object(SegmentationConfigurer, "configure_samples_per_gpu")
        mock_cfg_fp16_optimizer = mocker.patch.object(SegmentationConfigurer, "configure_fp16_optimizer")
        mock_cfg_compat_cfg = mocker.patch.object(SegmentationConfigurer, "configure_compat_cfg")

        model_cfg = copy.deepcopy(self.model_cfg)
        data_cfg = copy.deepcopy(self.data_cfg)
        returned_value = self.configurer.configure(model_cfg, "", data_cfg, True)
        mock_cfg_base.assert_called_once_with(model_cfg, data_cfg, None, None)
        mock_cfg_device.assert_called_once_with(model_cfg, True)
        mock_cfg_model.assert_called_once_with(model_cfg, None)
        mock_cfg_ckpt.assert_called_once_with(model_cfg, "")
        mock_cfg_task.assert_called_once_with(model_cfg, True)
        mock_cfg_hook.assert_called_once_with(model_cfg)
        mock_cfg_gpu.assert_called_once_with(model_cfg, "train")
        mock_cfg_fp16_optimizer.assert_called_once_with(model_cfg)
        mock_cfg_compat_cfg.assert_called_once_with(model_cfg)
        assert returned_value == model_cfg

    @e2e_pytest_unit
    def test_configure_base(self, mocker):
        mocker.patch(
            "otx.algorithms.segmentation.adapters.mmseg.configurer.align_data_config_with_recipe",
            return_value=True,
        )

        model_cfg = copy.deepcopy(self.model_cfg)
        data_cfg = copy.deepcopy(self.data_cfg._cfg_dict)
        self.configurer.configure_base(model_cfg, data_cfg, [], [])

    @e2e_pytest_unit
    def test_configure_model(self):
        ir_options = {"ir_model_path": {"ir_weight_path": "", "ir_weight_init": ""}}
        self.configurer.configure_model(self.model_cfg, ir_options)
        assert self.model_cfg.model_task

    @e2e_pytest_unit
    def test_configure_data(self, mocker):
        data_cfg = copy.deepcopy(self.data_cfg)
        self.configurer.configure_data(self.model_cfg, True, data_cfg)
        assert self.model_cfg.data
        assert self.model_cfg.data.train
        assert self.model_cfg.data.val

    @e2e_pytest_unit
    def test_configure_task(self, mocker):
        model_cfg = copy.deepcopy(self.model_cfg)
        mock_cfg_classes = mocker.patch.object(SegmentationConfigurer, "configure_classes")
        mock_cfg_ignore = mocker.patch.object(SegmentationConfigurer, "configure_decode_head")
        self.configurer.configure_task(model_cfg, True)

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
    def test_configure_device(self, mocker):
        mocker.patch(
            "torch.distributed.is_initialized",
            return_value=True,
        )
        world_size = 2
        mocker.patch.object(configurer, "dist").get_world_size.return_value = world_size
        mocker.patch("os.environ", return_value={"LOCAL_RANK": 2})
        config = copy.deepcopy(self.model_cfg)
        origin_lr = config.optimizer.lr
        self.configurer.configure_device(config, True)
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
        self.configurer.configure_device(config, True)
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
        self.configurer.configure_device(config, True)
        assert config.distributed is False
        assert config.device == "cuda"

    @e2e_pytest_unit
    def test_configure_samples_per_gpu(self):
        model_cfg = copy.deepcopy(self.model_cfg)
        model_cfg.data.train.dataset.otx_dataset = range(1)
        self.configurer.configure_samples_per_gpu(model_cfg, "train")
        assert model_cfg.data.train_dataloader == {"samples_per_gpu": 1, "drop_last": True}

    @e2e_pytest_unit
    def test_configure_fp16_optimizer(self):
        model_cfg = copy.deepcopy(self.model_cfg)
        model_cfg.fp16 = {}
        self.configurer.configure_fp16_optimizer(model_cfg)
        assert model_cfg.optimizer_config.type == "Fp16OptimizerHook"

        model_cfg.fp16 = {}
        model_cfg.optimizer_config.type = "SAMOptimizerHook"
        self.configurer.configure_fp16_optimizer(model_cfg)
        assert model_cfg.optimizer_config.type == "Fp16SAMOptimizerHook"

        model_cfg.fp16 = {}
        model_cfg.optimizer_config.type = "DummyOptimizerHook"
        self.configurer.configure_fp16_optimizer(model_cfg)
        assert model_cfg.optimizer_config.type == "DummyOptimizerHook"

    @e2e_pytest_unit
    def test_configure_compat_cfg(self):
        model_cfg = copy.deepcopy(self.model_cfg)
        model_cfg.data.train_dataloader = {}
        model_cfg.data.val_dataloader = {}
        model_cfg.data.test_dataloader = {}
        self.configurer.configure_compat_cfg(model_cfg)

    @e2e_pytest_unit
    def test_configure_hook(self):
        model_cfg = copy.deepcopy(self.model_cfg)
        model_cfg.custom_hook_options = {"LazyEarlyStoppingHook": {"start": 5}, "LoggerReplaceHook": {"_delete_": True}}
        self.configurer.configure_hook(model_cfg)
        assert model_cfg.custom_hooks[0]["start"] == 5


class TestIncrSegmentationConfigurer:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.configurer = IncrSegmentationConfigurer()
        self.model_cfg = MPAConfig.fromfile(os.path.join(DEFAULT_SEG_TEMPLATE_DIR, "model.py"))
        self.data_cfg = MPAConfig.fromfile(os.path.join(DEFAULT_SEG_TEMPLATE_DIR, "data_pipeline.py"))

    @e2e_pytest_unit
    def test_configure_task(self, mocker):
        mocker.patch.object(SegmentationConfigurer, "configure_task")
        self.configurer.configure_task(self.model_cfg, True)
        assert self.model_cfg.custom_hooks[1].type == "TaskAdaptHook"
        assert self.model_cfg.custom_hooks[1].sampler_flag is False


class TestSemiSLSegmentationConfigurer:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.configurer = SemiSLSegmentationConfigurer()
        self.model_cfg = MPAConfig.fromfile(os.path.join(DEFAULT_SEG_TEMPLATE_DIR, "model.py"))
        self.data_cfg = MPAConfig.fromfile(os.path.join(DEFAULT_SEG_TEMPLATE_DIR, "data_pipeline.py"))

    @e2e_pytest_unit
    def test_configure_data(self, mocker):
        data_cfg = copy.deepcopy(self.data_cfg)
        unlabeled_dict = dict(data=dict(unlabeled=dict(otx_dataset="foo")))
        data_cfg.merge_from_dict(unlabeled_dict)
        mock_ul_dataloader = mocker.patch.object(SemiSLSegmentationConfigurer, "configure_unlabeled_dataloader")
        self.configurer.configure_data(self.model_cfg, True, data_cfg)
        mock_ul_dataloader.assert_called_once()

    @e2e_pytest_unit
    def test_configure_task(self, mocker):
        model_cfg = ConfigDict(dict(model=dict(type="", task_adapt=True)))
        mock_remove_hook = mocker.patch("otx.algorithms.segmentation.adapters.mmseg.configurer.remove_custom_hook")
        self.configurer.configure_task(model_cfg, True)
        mock_remove_hook.assert_called_once()

    @e2e_pytest_unit
    def test_configure_unlabeled_dataloader(self, mocker):
        data_cfg = copy.deepcopy(self.data_cfg)
        data_cfg.model_task = "segmentation"
        data_cfg.distributed = False
        unlabeled_dict = dict(data=dict(unlabeled=dict(otx_dataset="foo")))
        data_cfg.merge_from_dict(unlabeled_dict)
        mocker_build_dataset = mocker.patch("otx.algorithms.segmentation.adapters.mmseg.configurer.build_dataset")
        mocker_build_dataloader = mocker.patch("otx.algorithms.segmentation.adapters.mmseg.configurer.build_dataloader")
        self.configurer.configure_data(self.model_cfg, True, data_cfg)

        mocker_build_dataset.assert_called_once()
        mocker_build_dataloader.assert_called_once()
        assert "ComposedDataLoadersHook" in [hook["type"] for hook in self.model_cfg.custom_hooks]
