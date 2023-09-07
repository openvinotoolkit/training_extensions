import copy
import os

import pytest
import tempfile
from mmcv.runner import CheckpointLoader
from mmcv.utils import ConfigDict

from otx.algorithms.common.adapters.mmcv.utils.config_utils import OTXConfig
from otx.algorithms.classification.adapters.mmcls import configurer
from otx.algorithms.classification.adapters.mmcls.configurer import (
    ClassificationConfigurer,
    IncrClassificationConfigurer,
    SemiSLClassificationConfigurer,
)
from otx.algorithms.common.configs.configuration_enums import InputSizePreset
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.algorithms.classification.test_helper import DEFAULT_CLS_TEMPLATE_DIR


class TestClassificationConfigurer:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.configurer = ClassificationConfigurer("classification", True)
        self.model_cfg = OTXConfig.fromfile(os.path.join(DEFAULT_CLS_TEMPLATE_DIR, "model.py"))
        data_pipeline_cfg = OTXConfig.fromfile(os.path.join(DEFAULT_CLS_TEMPLATE_DIR, "data_pipeline.py"))
        self.model_cfg.merge_from_dict(data_pipeline_cfg)
        self.data_cfg = OTXConfig(
            {
                "data": {
                    "train": {"otx_dataset": [], "labels": []},
                    "val": {"otx_dataset": [], "labels": []},
                    "test": {"otx_dataset": [], "labels": []},
                }
            }
        )

        self.multilabel_model_cfg = OTXConfig.fromfile(os.path.join(DEFAULT_CLS_TEMPLATE_DIR, "model_multilabel.py"))
        self.hierarchical_model_cfg = OTXConfig.fromfile(
            os.path.join(DEFAULT_CLS_TEMPLATE_DIR, "model_hierarchical.py")
        )

    @e2e_pytest_unit
    def test_configure(self, mocker):
        mock_cfg_base = mocker.patch.object(ClassificationConfigurer, "configure_base")
        mock_cfg_device = mocker.patch.object(ClassificationConfigurer, "configure_device")
        mock_cfg_ckpt = mocker.patch.object(ClassificationConfigurer, "configure_ckpt")
        mock_cfg_model = mocker.patch.object(ClassificationConfigurer, "configure_model")
        mock_cfg_data = mocker.patch.object(ClassificationConfigurer, "configure_data")
        mock_cfg_task = mocker.patch.object(ClassificationConfigurer, "configure_task")
        mock_cfg_gpu = mocker.patch.object(ClassificationConfigurer, "configure_samples_per_gpu")
        mock_cfg_fp16 = mocker.patch.object(ClassificationConfigurer, "configure_fp16")
        mock_cfg_compat_cfg = mocker.patch.object(ClassificationConfigurer, "configure_compat_cfg")
        mock_cfg_input_size = mocker.patch.object(ClassificationConfigurer, "configure_input_size")

        model_cfg = copy.deepcopy(self.model_cfg)
        data_cfg = copy.deepcopy(self.data_cfg)
        returned_value = self.configurer.configure(model_cfg, "", data_cfg)
        mock_cfg_base.assert_called_once_with(model_cfg, data_cfg, None, None)
        mock_cfg_device.assert_called_once_with(model_cfg)
        mock_cfg_ckpt.assert_called_once_with(model_cfg, "")
        mock_cfg_model.assert_called_once_with(model_cfg, None)
        mock_cfg_data.assert_called_once_with(model_cfg, data_cfg)
        mock_cfg_task.assert_called_once_with(model_cfg)
        mock_cfg_gpu.assert_called_once_with(model_cfg)
        mock_cfg_fp16.assert_called_once_with(model_cfg)
        mock_cfg_compat_cfg.assert_called_once_with(model_cfg)
        mock_cfg_input_size.assert_called_once_with(model_cfg, InputSizePreset.DEFAULT, "")
        assert returned_value == model_cfg

    @e2e_pytest_unit
    def test_configure_base(self, mocker):
        model_cfg = copy.deepcopy(self.model_cfg)
        data_cfg = copy.deepcopy(self.data_cfg._cfg_dict)
        self.configurer.configure_base(model_cfg, data_cfg, [], [])

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
    def test_configure_model(self):
        ir_options = {"ir_model_path": {"ir_weight_path": "", "ir_weight_init": ""}}
        self.model_cfg.model.head.in_channels = -1
        self.configurer.configure_model(self.model_cfg, ir_options)
        assert self.model_cfg.model_task
        assert self.model_cfg.model.head.in_channels == 960

        multilabel_model_cfg = self.multilabel_model_cfg
        self.configurer.configure_model(multilabel_model_cfg, ir_options)

        h_label_model_cfg = self.hierarchical_model_cfg
        self.configurer.configure_model(h_label_model_cfg, ir_options)

    @e2e_pytest_unit
    def test_configure_model_not_classification_task(self):
        ir_options = {"ir_model_path": {"ir_weight_path": "", "ir_weight_init": ""}}
        configure_cfg = copy.deepcopy(self.model_cfg)
        configure_cfg.model.task = "detection"
        with pytest.raises(ValueError):
            self.configurer.configure_model(configure_cfg, ir_options)

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

    @e2e_pytest_unit
    def test_configure_data(self, mocker):
        data_cfg = copy.deepcopy(self.data_cfg)
        self.configurer.configure_data(self.model_cfg, data_cfg)
        assert self.model_cfg.data
        assert self.model_cfg.data.train
        assert self.model_cfg.data.val

    @e2e_pytest_unit
    def test_configure_task(self, mocker):
        model_cfg = copy.deepcopy(self.model_cfg)
        model_cfg.update(self.data_cfg)
        model_cfg.task_adapt = {"type": "default_task_adapt", "op": "REPLACE", "use_adaptive_anchor": True}
        self.configurer.configure_task(model_cfg)

        self.configurer.model_classes = []
        self.configurer.data_classes = ["red", "green"]
        self.configurer.configure_task(model_cfg)

    @e2e_pytest_unit
    def test_configure_samples_per_gpu(self):
        model_cfg = copy.deepcopy(self.model_cfg)
        model_cfg.data.train_dataloader = ConfigDict({"samples_per_gpu": 2})
        model_cfg.data.train.otx_dataset = range(1)
        self.configurer.configure_samples_per_gpu(model_cfg)
        assert model_cfg.data.train_dataloader == {"samples_per_gpu": 1, "drop_last": True}

    @e2e_pytest_unit
    def test_configure_fp16(self):
        model_cfg = copy.deepcopy(self.model_cfg)
        model_cfg.fp16 = {}
        model_cfg.optimizer_config.type = "OptimizerHook"
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
    def test_configure_compat_cfg(self):
        model_cfg = copy.deepcopy(self.model_cfg)
        model_cfg.update(self.data_cfg)
        model_cfg.data.train_dataloader = {}
        model_cfg.data.val_dataloader = {}
        model_cfg.data.test_dataloader = {}
        self.configurer.configure_compat_cfg(model_cfg)

    @e2e_pytest_unit
    def test_get_data_cfg(self):
        config = copy.deepcopy(self.model_cfg)
        config.update(self.data_cfg)
        config.data.train.dataset = ConfigDict({"dataset": [1, 2, 3]})
        assert [1, 2, 3] == self.configurer.get_data_cfg(config, "train")

    @e2e_pytest_unit
    @pytest.mark.parametrize("input_size", [None, (128, 128)])
    def test_configure_input_size(self, mocker, input_size):
        # prepare
        mock_cfg = mocker.MagicMock()
        mocker.patch.object(configurer, "get_configured_input_size", return_value=input_size)
        mock_input_manager = mocker.MagicMock()
        mock_input_manager_cls = mocker.patch.object(configurer, "InputSizeManager")
        mock_input_manager_cls.return_value = mock_input_manager

        # excute
        self.configurer.configure_input_size(mock_cfg, InputSizePreset.DEFAULT, self.data_cfg)

        # check
        if input_size is not None:
            mock_input_manager_cls.assert_called_once_with(mock_cfg.data)
            mock_input_manager.set_input_size.assert_called_once_with(input_size)
        else:
            mock_input_manager_cls.assert_not_called()


class TestIncrClassificationConfigurer:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.configurer = IncrClassificationConfigurer("classification", True)
        self.model_cfg = OTXConfig.fromfile(os.path.join(DEFAULT_CLS_TEMPLATE_DIR, "model.py"))
        self.data_cfg = OTXConfig.fromfile(os.path.join(DEFAULT_CLS_TEMPLATE_DIR, "data_pipeline.py"))

    def test_configure_task(self, mocker):
        mocker.patch.object(ClassificationConfigurer, "configure_task")
        self.model_cfg.update(self.data_cfg)
        self.model_cfg.task_adapt = {}
        self.configurer.task_adapt_type = "default_task_adapt"
        self.configurer.configure_task(self.model_cfg)
        assert "TaskAdaptHook" in [i.type for i in self.model_cfg.custom_hooks]


class TestSemiSLClassificationConfigurer:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.configurer = SemiSLClassificationConfigurer("classification", True)
        self.cfg = OTXConfig.fromfile(os.path.join(DEFAULT_CLS_TEMPLATE_DIR, "semisl", "model.py"))
        data_cfg = OTXConfig.fromfile(os.path.join(DEFAULT_CLS_TEMPLATE_DIR, "semisl", "data_pipeline.py"))
        self.cfg.merge_from_dict(data_cfg)

    @e2e_pytest_unit
    def test_configure_data(self, mocker):
        mocker.patch("otx.algorithms.common.adapters.mmcv.semisl_mixin.build_dataset", return_value=True)
        mocker.patch("otx.algorithms.common.adapters.mmcv.semisl_mixin.build_dataloader", return_value=True)

        data_cfg = OTXConfig(
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
