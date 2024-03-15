"""Test otx mmdet configurer."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import copy
import os

import pytest
import tempfile
from mmcv.runner import CheckpointLoader
from mmcv.utils import ConfigDict

from otx.api.entities.model_template import TaskType
from otx.algorithms.common.adapters.mmcv.utils.config_utils import OTXConfig
from otx.algorithms.detection.adapters.mmdet import configurer
from otx.algorithms.detection.adapters.mmdet.configurer import (
    DetectionConfigurer,
    IncrDetectionConfigurer,
    SemiSLDetectionConfigurer,
)
from otx.algorithms.common.configs.configuration_enums import InputSizePreset
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.algorithms.detection.test_helpers import (
    DEFAULT_DET_TEMPLATE_DIR,
    generate_det_dataset,
)


@pytest.fixture
def device_availability_func(mocker):
    return {
        "cuda": mocker.patch("torch.cuda.is_available"),
        "xpu": mocker.patch("otx.algorithms.common.adapters.mmcv.configurer.is_xpu_available"),
        "hpu": mocker.patch("otx.algorithms.common.adapters.mmcv.configurer.is_hpu_available"),
    }


class TestDetectionConfigurer:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.configurer = DetectionConfigurer(
            "segmentation",
            True,
            False,
            {},
            None,
            None,
            None,
        )
        self.model_cfg = OTXConfig.fromfile(os.path.join(DEFAULT_DET_TEMPLATE_DIR, "model.py"))
        self.data_pipeline_path = os.path.join(DEFAULT_DET_TEMPLATE_DIR, "data_pipeline.py")

        self.det_dataset, self.det_labels = generate_det_dataset(TaskType.DETECTION, 100)
        self.data_cfg = ConfigDict(
            {
                "data": {
                    "train": {"otx_dataset": self.det_dataset, "labels": self.det_labels},
                    "val": {"otx_dataset": self.det_dataset, "labels": self.det_labels},
                    "test": {"otx_dataset": self.det_dataset, "labels": self.det_labels},
                }
            }
        )

    @e2e_pytest_unit
    def test_configure(self, mocker):
        mock_cfg_merge = mocker.patch.object(DetectionConfigurer, "merge_configs")
        mock_cfg_ckpt = mocker.patch.object(DetectionConfigurer, "configure_ckpt")
        mock_cfg_env = mocker.patch.object(DetectionConfigurer, "configure_env")
        mock_cfg_data_pipeline = mocker.patch.object(DetectionConfigurer, "configure_data_pipeline")
        mock_cfg_recipe = mocker.patch.object(DetectionConfigurer, "configure_recipe")
        mock_cfg_model = mocker.patch.object(DetectionConfigurer, "configure_model")
        mock_cfg_hook = mocker.patch.object(DetectionConfigurer, "configure_hooks")
        mock_cfg_compat_cfg = mocker.patch.object(DetectionConfigurer, "configure_compat_cfg")

        model_cfg = copy.deepcopy(self.model_cfg)
        model_cfg.model_task = "detection"
        data_cfg = copy.deepcopy(self.data_cfg)
        returned_value = self.configurer.configure(
            model_cfg,
            self.data_pipeline_path,
            None,
            "",
            data_cfg,
            train_dataset=self.det_dataset,
            max_num_detections=100,
        )

        mock_cfg_merge.assert_called_once_with(
            model_cfg,
            data_cfg,
            self.data_pipeline_path,
            None,
            train_dataset=self.det_dataset,
            max_num_detections=100,
        )
        mock_cfg_ckpt.assert_called_once_with(model_cfg, "")
        mock_cfg_env.assert_called_once_with(model_cfg)
        mock_cfg_data_pipeline.assert_called_once_with(
            model_cfg, None, "", train_dataset=self.det_dataset, max_num_detections=100
        )
        mock_cfg_recipe.assert_called_once_with(model_cfg, train_dataset=self.det_dataset, max_num_detections=100)
        mock_cfg_hook.assert_called_once_with(model_cfg)
        mock_cfg_model.assert_called_once_with(
            model_cfg, None, None, None, train_dataset=self.det_dataset, max_num_detections=100
        )
        mock_cfg_compat_cfg.assert_called_once_with(model_cfg)
        assert returned_value == model_cfg

    @e2e_pytest_unit
    def test_merge_configs(self, mocker):
        mocker.patch("otx.algorithms.common.adapters.mmcv.configurer.patch_from_hyperparams", return_value=True)
        mocker.patch("otx.algorithms.detection.adapters.mmdet.configurer.patch_tiling", return_value=True)
        self.configurer.merge_configs(self.model_cfg, self.data_cfg, self.data_pipeline_path, None)
        assert self.model_cfg.data
        assert self.model_cfg.data.train
        assert self.model_cfg.data.val

    @e2e_pytest_unit
    def test_configure_ckpt(self, mocker):
        model_cfg = copy.deepcopy(self.model_cfg)
        model_cfg.resume = True

        mocker.patch.object(
            CheckpointLoader,
            "load_checkpoint",
            return_value={"model": None},
        )
        with tempfile.TemporaryDirectory() as tempdir:
            self.configurer.configure_ckpt(model_cfg, os.path.join(tempdir, "dummy.pth"))
        for hook in model_cfg.custom_hooks:
            if hook.type in self.configurer.ema_hooks:
                assert hook.resume_from == model_cfg.resume_from

    @e2e_pytest_unit
    def test_configure_env(self):
        data_pipeline_cfg = OTXConfig.fromfile(self.data_pipeline_path)
        self.model_cfg.merge_from_dict(data_pipeline_cfg)
        self.configurer.configure_env(self.model_cfg)

    @e2e_pytest_unit
    @pytest.mark.parametrize("current_device", ["cpu", "cuda", "xpu", "hpu"])
    def test_configure_device(self, mocker, device_availability_func, current_device):
        for key, mock_func in device_availability_func.items():
            if current_device == key:
                mock_func.return_value = True
            else:
                mock_func.return_value = False

        mocker.patch(
            "torch.distributed.is_initialized",
            return_value=False,
        )

        config = copy.deepcopy(self.model_cfg)
        self.configurer.configure_device(config)
        assert config.distributed is False
        assert config.device == current_device

    @e2e_pytest_unit
    def test_configure_dist_device(self, mocker):
        mocker.patch(
            "torch.distributed.is_initialized",
            return_value=True,
        )

        config = copy.deepcopy(self.model_cfg)
        mocker.patch("torch.distributed.get_world_size", return_value=2)
        world_size = 2
        mocker.patch("os.environ", return_value={"LOCAL_RANK": 2})
        origin_lr = config.optimizer.lr

        self.configurer.configure_device(config)
        assert config.distributed is True
        assert config.optimizer.lr == pytest.approx(origin_lr * world_size)

    @e2e_pytest_unit
    def test_configure_samples_per_gpu(self):
        model_cfg = copy.deepcopy(self.model_cfg)
        data_pipeline_cfg = OTXConfig.fromfile(self.data_pipeline_path)
        model_cfg.merge_from_dict(data_pipeline_cfg)
        model_cfg.data.train_dataloader = ConfigDict({"samples_per_gpu": 2})
        model_cfg.data.train.otx_dataset = range(1)
        self.configurer.configure_samples_per_gpu(model_cfg)
        assert model_cfg.data.train_dataloader == {"samples_per_gpu": 1, "drop_last": True}

    @e2e_pytest_unit
    @pytest.mark.parametrize("input_size", [None, (0, 0), (256, 256)])
    @pytest.mark.parametrize("training", [True, False])
    def test_configure_input_size_not_yolox(self, mocker, input_size, training):
        # prepare
        mock_cfg = mocker.MagicMock()
        mock_input_manager_cls = mocker.patch.object(configurer, "InputSizeManager")
        mock_input_manager = mock_input_manager_cls.return_value
        mock_input_manager.get_trained_input_size.return_value = (32, 32)
        mock_input_manager_cls.return_value = mock_input_manager
        mock_base_configurer_cls = mocker.patch.object(configurer, "BaseConfigurer")
        mock_base_configurer_cls.adapt_input_size_to_dataset.return_value = (64, 64)

        # execute
        self.configurer.configure_input_size(mock_cfg, input_size, "ckpt/path", training=training)

        # check
        if input_size is None:
            mock_input_manager.set_input_size.assert_not_called()
        elif input_size == (0, 0):
            if training:
                mock_input_manager.set_input_size.assert_called_once_with((64, 64))
            else:
                mock_input_manager.set_input_size.assert_called_once_with((32, 32))
        else:
            mock_input_manager.set_input_size.assert_called_once_with(input_size)

    @e2e_pytest_unit
    @pytest.mark.parametrize("is_yolox_tiny", [True, False])
    def test_configure_input_size_yolox(self, mocker, is_yolox_tiny):
        # prepare
        mock_cfg = mocker.MagicMock()
        mock_cfg.model.type = "CustomYOLOX"
        if is_yolox_tiny:
            mock_cfg.model.backbone.widen_factor = 0.375
            base_input_size = {
                "train": (640, 640),
                "val": (416, 416),
                "test": (416, 416),
                "unlabeled": (992, 736),
            }
        else:
            base_input_size = None

        mock_input_manager_cls = mocker.patch.object(configurer, "InputSizeManager")
        mock_input_manager = mock_input_manager_cls.return_value
        mock_input_manager.get_configured_input_size.return_value = None

        # excute
        self.configurer.configure_input_size(mock_cfg, InputSizePreset.DEFAULT)

        # check
        mock_input_manager_cls.assert_called_once_with(mock_cfg, base_input_size)

    @e2e_pytest_unit
    @pytest.mark.parametrize("optimizer_hook", ["OptimizerHook", "SAMOptimizerHook", "DummyOptimizerHook"])
    def test_configure_fp16_cpu(self, device_availability_func, optimizer_hook):
        for func in device_availability_func.values():
            func.return_value = False

        model_cfg = copy.deepcopy(self.model_cfg)
        model_cfg.fp16 = {}
        model_cfg.optimizer_config.type = optimizer_hook
        self.configurer.configure_fp16(model_cfg)
        assert model_cfg.optimizer_config.type == optimizer_hook

    @e2e_pytest_unit
    @pytest.mark.parametrize("optimizer_hook", ["OptimizerHook", "SAMOptimizerHook", "DummyOptimizerHook"])
    def test_configure_fp16_cuda(self, device_availability_func, optimizer_hook):
        for key, func in device_availability_func.items():
            if key == "cuda":
                func.return_value = True
            else:
                func.return_value = False

        if "Dummy" in optimizer_hook:
            expected_optimizer_hook = optimizer_hook
        else:
            expected_optimizer_hook = f"Fp16{optimizer_hook}"

        model_cfg = copy.deepcopy(self.model_cfg)
        model_cfg.fp16 = {}
        model_cfg.optimizer_config.type = optimizer_hook
        self.configurer.configure_fp16(model_cfg)
        assert model_cfg.optimizer_config.type == expected_optimizer_hook

    @e2e_pytest_unit
    def test_configure_model(self):
        ir_options = {"ir_model_path": {"ir_weight_path": "", "ir_weight_init": ""}}
        data_pipeline_cfg = OTXConfig.fromfile(self.data_pipeline_path)
        self.model_cfg.merge_from_dict(data_pipeline_cfg)
        self.model_cfg.merge_from_dict(self.data_cfg)
        self.configurer.configure_model(self.model_cfg, [], self.det_labels, ir_options, train_dataset=self.det_dataset)
        assert len(self.configurer.model_classes) == 3

    @e2e_pytest_unit
    def test_configure_model_without_model(self):
        ir_options = {"ir_model_path": {"ir_weight_path": "", "ir_weight_init": ""}}
        data_pipeline_cfg = OTXConfig.fromfile(self.data_pipeline_path)
        self.model_cfg.merge_from_dict(data_pipeline_cfg)
        self.model_cfg.merge_from_dict(self.data_cfg)
        model_cfg = copy.deepcopy(self.model_cfg)
        model_cfg.pop("model")
        with pytest.raises(AttributeError):
            self.configurer.configure_model(model_cfg, [], self.det_labels, ir_options, train_dataset=self.det_dataset)

    @e2e_pytest_unit
    def test_configure_task(self, mocker):
        ssd_dir = os.path.join("src/otx/algorithms/detection/configs/detection", "mobilenetv2_ssd")
        ssd_cfg = OTXConfig.fromfile(os.path.join(ssd_dir, "model.py"))
        data_pipeline_cfg = OTXConfig.fromfile(os.path.join(ssd_dir, "data_pipeline.py"))
        ssd_cfg.task_adapt = {"type": "default_task_adapt", "op": "REPLACE", "use_adaptive_anchor": True}
        model_cfg = copy.deepcopy(ssd_cfg)
        model_cfg.merge_from_dict(data_pipeline_cfg)
        self.configurer.configure_task(model_cfg, train_dataset=self.det_dataset)
        assert model_cfg.model.bbox_head.anchor_generator != ssd_cfg.model.bbox_head.anchor_generator

        model_cfg = copy.deepcopy(self.model_cfg)
        data_pipeline_cfg = OTXConfig.fromfile(self.data_pipeline_path)
        model_cfg.merge_from_dict(data_pipeline_cfg)
        model_cfg.task_adapt = {"type": "default_task_adapt", "op": "REPLACE", "use_adaptive_anchor": True}
        model_cfg.model.bbox_head.type = "ATSSHead"
        self.configurer.configure_task(model_cfg, train_dataset=self.det_dataset)

        model_cfg.model.bbox_head.type = "VFNetHead"
        self.configurer.configure_task(model_cfg, train_dataset=self.det_dataset)

        model_cfg.model.bbox_head.type = "YOLOXHead"
        model_cfg.data.train.type = "MultiImageMixDataset"
        self.configurer.configure_task(model_cfg, train_dataset=self.det_dataset)

        def mock_configure_classes(*args, **kwargs):
            return True

        mocker.patch.object(DetectionConfigurer, "configure_classes")
        self.configurer.model_classes = []
        self.configurer.data_classes = ["red", "green"]
        self.configurer.configure_classes = mock_configure_classes
        self.configurer.configure_task(model_cfg, train_dataset=self.det_dataset)

    @e2e_pytest_unit
    def test_configure_regularization(self):
        configure_cfg = copy.deepcopy(self.model_cfg)
        configure_cfg.model.l2sp_weight = 1.0
        self.configurer.configure_regularization(configure_cfg)
        assert "l2sp_ckpt" in configure_cfg.model
        assert configure_cfg.optimizer.weight_decay == 0.0

    @e2e_pytest_unit
    def test_configure_hooks(self):
        self.configurer.override_configs = {"custom_hooks": [{"type": "LazyEarlyStoppingHook", "patience": 6}]}
        self.configurer.time_monitor = []
        self.configurer.configure_hooks(self.model_cfg)
        assert self.model_cfg.custom_hooks[0]["patience"] == 6
        assert self.model_cfg.custom_hooks[-2]["type"] == "CancelInterfaceHook"
        assert self.model_cfg.custom_hooks[-1]["type"] == "OTXProgressHook"
        assert self.model_cfg.log_config.hooks[-1]["type"] == "OTXLoggerHook"

    @e2e_pytest_unit
    def test_configure_compat_cfg(self):
        model_cfg = copy.deepcopy(self.model_cfg)
        data_pipeline_cfg = OTXConfig.fromfile(self.data_pipeline_path)
        model_cfg.merge_from_dict(data_pipeline_cfg)
        model_cfg.data.train_dataloader = {}
        model_cfg.data.val_dataloader = {}
        model_cfg.data.test_dataloader = {}
        self.configurer.configure_compat_cfg(model_cfg)

    @e2e_pytest_unit
    def test_get_subset_data_cfg(self):
        config = copy.deepcopy(self.model_cfg)
        data_pipeline_cfg = OTXConfig.fromfile(self.data_pipeline_path)
        config.merge_from_dict(data_pipeline_cfg)
        config.data.train.dataset = ConfigDict({"dataset": [1, 2, 3]})
        assert [1, 2, 3] == self.configurer.get_subset_data_cfg(config, "train")


class TestIncrDetectionConfigurer:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.configurer = IncrDetectionConfigurer(
            "segmentation",
            True,
            False,
            {},
            None,
            None,
            None,
        )
        self.model_cfg = OTXConfig.fromfile(os.path.join(DEFAULT_DET_TEMPLATE_DIR, "model.py"))
        self.data_cfg = OTXConfig.fromfile(os.path.join(DEFAULT_DET_TEMPLATE_DIR, "data_pipeline.py"))
        self.det_dataset, self.det_labels = generate_det_dataset(TaskType.DETECTION, 100)

    def test_configure_task(self, mocker):
        mocker.patch.object(DetectionConfigurer, "configure_task")
        self.model_cfg.task_adapt = {}
        self.configurer.task_adapt_type = "default_task_adapt"
        self.configurer.configure_task(self.model_cfg, train_dataset=self.det_dataset)
        assert self.model_cfg.custom_hooks[2].type == "TaskAdaptHook"
        assert self.model_cfg.custom_hooks[2].sampler_flag is False


class TestSemiSLDetectionConfigurer:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.configurer = SemiSLDetectionConfigurer(
            "segmentation",
            True,
            False,
            {},
            None,
            None,
            None,
        )
        self.model_cfg = OTXConfig.fromfile(os.path.join(DEFAULT_DET_TEMPLATE_DIR, "semisl", "model.py"))
        self.data_cfg = OTXConfig.fromfile(os.path.join(DEFAULT_DET_TEMPLATE_DIR, "semisl", "data_pipeline.py"))
        self.model_cfg.merge_from_dict(self.data_cfg)
        self.det_dataset, self.det_labels = generate_det_dataset(TaskType.DETECTION, 100)

    @e2e_pytest_unit
    def test_configure_data_pipeline(self, mocker):
        mocker.patch("otx.algorithms.common.adapters.mmcv.semisl_mixin.build_dataset", return_value=True)
        mocker.patch("otx.algorithms.common.adapters.mmcv.semisl_mixin.build_dataloader", return_value=True)
        mocker.patch.object(DetectionConfigurer, "configure_input_size", return_value=True)

        data_cfg = OTXConfig(
            {
                "data": {
                    "train": {"otx_dataset": [], "labels": []},
                    "val": {"otx_dataset": [], "labels": []},
                    "test": {"otx_dataset": [], "labels": []},
                    "unlabeled": {"otx_dataset": self.det_dataset, "labels": []},
                }
            }
        )
        self.model_cfg.merge_from_dict(data_cfg)
        self.model_cfg.model_task = "detection"
        self.model_cfg.distributed = False
        self.configurer.configure_data_pipeline(self.model_cfg, InputSizePreset.DEFAULT, "")
        assert self.model_cfg.custom_hooks[-1]["type"] == "ComposedDataLoadersHook"
