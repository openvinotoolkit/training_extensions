import copy
import os
from otx.algorithms.common.utils.utils import is_xpu_available

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
        self.configurer = ClassificationConfigurer(
            "classification",
            True,
            False,
            {},
            None,
            None,
            None,
        )
        self.model_cfg = OTXConfig.fromfile(os.path.join(DEFAULT_CLS_TEMPLATE_DIR, "model.py"))
        self.data_pipeline_path = os.path.join(DEFAULT_CLS_TEMPLATE_DIR, "data_pipeline.py")
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
        mock_cfg_merge = mocker.patch.object(ClassificationConfigurer, "merge_configs")
        mock_cfg_ckpt = mocker.patch.object(ClassificationConfigurer, "configure_ckpt")
        mock_cfg_env = mocker.patch.object(ClassificationConfigurer, "configure_env")
        mock_cfg_data_pipeline = mocker.patch.object(ClassificationConfigurer, "configure_data_pipeline")
        mock_cfg_recipe = mocker.patch.object(ClassificationConfigurer, "configure_recipe")
        mock_cfg_model = mocker.patch.object(ClassificationConfigurer, "configure_model")
        mock_cfg_hook = mocker.patch.object(ClassificationConfigurer, "configure_hooks")
        mock_cfg_compat_cfg = mocker.patch.object(ClassificationConfigurer, "configure_compat_cfg")

        model_cfg = copy.deepcopy(self.model_cfg)
        model_cfg.model_task = "classification"
        data_cfg = copy.deepcopy(self.data_cfg)
        returned_value = self.configurer.configure(model_cfg, self.data_pipeline_path, None, "", data_cfg)

        mock_cfg_merge.assert_called_once_with(model_cfg, data_cfg, self.data_pipeline_path, None)
        mock_cfg_ckpt.assert_called_once_with(model_cfg, "")
        mock_cfg_env.assert_called_once_with(model_cfg)
        mock_cfg_data_pipeline.assert_called_once_with(model_cfg, None, "")
        mock_cfg_recipe.assert_called_once_with(model_cfg)
        mock_cfg_model.assert_called_once_with(model_cfg, None, None, None)
        mock_cfg_hook.assert_called_once_with(model_cfg)
        mock_cfg_compat_cfg.assert_called_once_with(model_cfg)
        assert returned_value == model_cfg

    @e2e_pytest_unit
    def test_merge_configs(self, mocker):
        mocker.patch("otx.algorithms.common.adapters.mmcv.configurer.patch_from_hyperparams", return_value=True)
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

    @e2e_pytest_unit
    def test_configure_env(self):
        data_pipeline_cfg = OTXConfig.fromfile(self.data_pipeline_path)
        self.model_cfg.merge_from_dict(data_pipeline_cfg)
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
        assert config.device in ["cpu", "xpu"]

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
        data_pipeline_cfg = OTXConfig.fromfile(self.data_pipeline_path)
        model_cfg.merge_from_dict(data_pipeline_cfg)
        model_cfg.data.train_dataloader = ConfigDict({"samples_per_gpu": 2})
        model_cfg.data.train.otx_dataset = range(1)
        self.configurer.configure_samples_per_gpu(model_cfg)
        assert model_cfg.data.train_dataloader == {"samples_per_gpu": 1, "drop_last": True}

    @e2e_pytest_unit
    @pytest.mark.parametrize("input_size", [None, (0, 0), (128, 128)])
    @pytest.mark.parametrize("training", [True, False])
    def test_configure_input_size(self, mocker, input_size, training):
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
    def test_configure_fp16(self):
        if is_xpu_available():
            pytest.skip("FP16 is not supported on XPU")
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
    def test_configure_model(self):
        ir_options = {"ir_model_path": {"ir_weight_path": "", "ir_weight_init": ""}}
        data_pipeline_cfg = OTXConfig.fromfile(self.data_pipeline_path)
        self.model_cfg.merge_from_dict(data_pipeline_cfg)
        self.model_cfg.model.head.in_channels = -1
        self.model_cfg.merge_from_dict(self.data_cfg)
        self.configurer.configure_model(self.model_cfg, [], [], ir_options)
        assert self.model_cfg.model_task
        assert self.model_cfg.model.head.in_channels == 960

        multilabel_model_cfg = self.multilabel_model_cfg
        multilabel_model_cfg.merge_from_dict(data_pipeline_cfg)
        multilabel_model_cfg.merge_from_dict(self.data_cfg)
        self.configurer.configure_model(multilabel_model_cfg, [], [], ir_options)

        h_label_model_cfg = self.hierarchical_model_cfg
        h_label_model_cfg.merge_from_dict(data_pipeline_cfg)
        h_label_model_cfg.merge_from_dict(self.data_cfg)
        self.configurer.configure_model(h_label_model_cfg, [], [], ir_options)

    @e2e_pytest_unit
    def test_configure_model_not_classification_task(self):
        ir_options = {"ir_model_path": {"ir_weight_path": "", "ir_weight_init": ""}}
        data_pipeline_cfg = OTXConfig.fromfile(self.data_pipeline_path)
        self.model_cfg.merge_from_dict(data_pipeline_cfg)
        self.model_cfg.merge_from_dict(self.data_cfg)
        configure_cfg = copy.deepcopy(self.model_cfg)
        configure_cfg.model.task = "detection"
        with pytest.raises(ValueError):
            self.configurer.configure_model(configure_cfg, [], [], ir_options)

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
        model_cfg.update(self.data_cfg)
        model_cfg.data.train_dataloader = {}
        model_cfg.data.val_dataloader = {}
        model_cfg.data.test_dataloader = {}
        self.configurer.configure_compat_cfg(model_cfg)

    @e2e_pytest_unit
    def test_get_subset_data_cfg(self):
        config = copy.deepcopy(self.model_cfg)
        config.update(self.data_cfg)
        config.data.train.dataset = ConfigDict({"dataset": [1, 2, 3]})
        assert [1, 2, 3] == self.configurer.get_subset_data_cfg(config, "train")


class TestIncrClassificationConfigurer:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.configurer = IncrClassificationConfigurer(
            "classification",
            True,
            False,
            {},
            None,
            None,
            None,
        )
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
        self.configurer = SemiSLClassificationConfigurer(
            "classification",
            True,
            False,
            {},
            None,
            None,
            None,
        )
        self.cfg = OTXConfig.fromfile(os.path.join(DEFAULT_CLS_TEMPLATE_DIR, "semisl", "model.py"))
        data_cfg = OTXConfig.fromfile(os.path.join(DEFAULT_CLS_TEMPLATE_DIR, "semisl", "data_pipeline.py"))
        self.cfg.merge_from_dict(data_cfg)

    @e2e_pytest_unit
    def test_configure_data_pipeline(self, mocker):
        mocker.patch("otx.algorithms.common.adapters.mmcv.semisl_mixin.build_dataset", return_value=True)
        mocker.patch("otx.algorithms.common.adapters.mmcv.semisl_mixin.build_dataloader", return_value=True)
        mocker.patch.object(ClassificationConfigurer, "configure_input_size", return_value=True)

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
        self.cfg.merge_from_dict(data_cfg)
        self.cfg.model_task = "classification"
        self.cfg.distributed = False
        self.configurer.configure_data_pipeline(self.cfg, InputSizePreset.DEFAULT, "")
        assert self.cfg.custom_hooks[-1]["type"] == "ComposedDataLoadersHook"
