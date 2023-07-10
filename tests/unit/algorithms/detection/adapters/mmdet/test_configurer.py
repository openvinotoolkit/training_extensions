import copy
import os

import pytest
import tempfile
from mmcv.utils import ConfigDict

from otx.api.entities.model_template import TaskType
from otx.algorithms.common.adapters.mmcv.utils.config_utils import MPAConfig
from otx.algorithms.detection.adapters.mmdet import configurer
from otx.algorithms.detection.adapters.mmdet.configurer import (
    DetectionConfigurer,
    IncrDetectionConfigurer,
    SemiSLDetectionConfigurer,
)
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.algorithms.detection.test_helpers import (
    DEFAULT_DET_TEMPLATE_DIR,
    generate_det_dataset,
)


class TestDetectionConfigurer:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.configurer = DetectionConfigurer()
        self.model_cfg = MPAConfig.fromfile(os.path.join(DEFAULT_DET_TEMPLATE_DIR, "model.py"))
        self.data_cfg = MPAConfig.fromfile(os.path.join(DEFAULT_DET_TEMPLATE_DIR, "data_pipeline.py"))
        self.det_dataset, self.det_labels = generate_det_dataset(TaskType.DETECTION, 100)

    @e2e_pytest_unit
    def test_configure(self, mocker):
        mock_cfg_base = mocker.patch.object(DetectionConfigurer, "configure_base")
        mock_cfg_device = mocker.patch.object(DetectionConfigurer, "configure_device")
        mock_cfg_model = mocker.patch.object(DetectionConfigurer, "configure_model")
        mock_cfg_ckpt = mocker.patch.object(DetectionConfigurer, "configure_ckpt")
        mock_cfg_regularization = mocker.patch.object(DetectionConfigurer, "configure_regularization")
        mock_cfg_task = mocker.patch.object(DetectionConfigurer, "configure_task")
        mock_cfg_hook = mocker.patch.object(DetectionConfigurer, "configure_hook")
        mock_cfg_gpu = mocker.patch.object(DetectionConfigurer, "configure_samples_per_gpu")
        mock_cfg_fp16_optimizer = mocker.patch.object(DetectionConfigurer, "configure_fp16_optimizer")
        mock_cfg_compat_cfg = mocker.patch.object(DetectionConfigurer, "configure_compat_cfg")

        model_cfg = copy.deepcopy(self.model_cfg)
        data_cfg = copy.deepcopy(self.data_cfg)
        returned_value = self.configurer.configure(model_cfg, self.det_dataset, "", data_cfg, True)
        mock_cfg_base.assert_called_once_with(model_cfg, data_cfg, None, None)
        mock_cfg_device.assert_called_once_with(model_cfg, True)
        mock_cfg_model.assert_called_once_with(model_cfg, None)
        mock_cfg_ckpt.assert_called_once_with(model_cfg, "")
        mock_cfg_regularization.assert_called_once_with(model_cfg, True)
        mock_cfg_task.assert_called_once_with(model_cfg, self.det_dataset, True)
        mock_cfg_hook.assert_called_once_with(model_cfg)
        mock_cfg_gpu.assert_called_once_with(model_cfg, "train")
        mock_cfg_fp16_optimizer.assert_called_once_with(model_cfg)
        mock_cfg_compat_cfg.assert_called_once_with(model_cfg)
        assert returned_value == model_cfg

    @e2e_pytest_unit
    def test_configure_base(self, mocker):
        mocker.patch(
            "otx.algorithms.detection.adapters.mmdet.configurer.align_data_config_with_recipe",
            return_value=True,
        )

        model_cfg = copy.deepcopy(self.model_cfg)
        data_cfg = copy.deepcopy(self.data_cfg._cfg_dict)
        self.configurer.configure_base(model_cfg, data_cfg, [], [])

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
    def test_configure_model(self):
        ir_options = {"ir_model_path": {"ir_weight_path": "", "ir_weight_init": ""}}
        self.configurer.configure_model(self.model_cfg, ir_options)
        assert self.model_cfg.model_task

    @e2e_pytest_unit
    def test_configure_ckpt(self, mocker):
        model_cfg = copy.deepcopy(self.model_cfg)
        model_cfg.resume = True

        mocker.patch(
            "otx.algorithms.detection.adapters.mmdet.configurer.CheckpointLoader.load_checkpoint",
            return_value={"model": None},
        )
        with tempfile.TemporaryDirectory() as tempdir:
            self.configurer.configure_ckpt(model_cfg, os.path.join(tempdir, "dummy.pth"))

    @e2e_pytest_unit
    def test_configure_model_without_model(self):
        ir_options = {"ir_model_path": {"ir_weight_path": "", "ir_weight_init": ""}}
        model_cfg = copy.deepcopy(self.model_cfg)
        model_cfg.pop("model")
        with pytest.raises(AttributeError):
            self.configurer.configure_model(model_cfg, ir_options)

    @e2e_pytest_unit
    def test_configure_model_not_detection_task(self):
        ir_options = {"ir_model_path": {"ir_weight_path": "", "ir_weight_init": ""}}
        configure_cfg = copy.deepcopy(self.model_cfg)
        configure_cfg.model.task = "classification"
        with pytest.raises(ValueError):
            self.configurer.configure_model(configure_cfg, ir_options)

    @e2e_pytest_unit
    def test_configure_data(self, mocker):
        data_cfg = copy.deepcopy(self.data_cfg)
        data_cfg.data.pipeline_options = dict(
            MinIouRandomCrop=dict(min_crop_size=0.1),
            Resize=dict(
                img_scale=[(1344, 480), (1344, 960)],
                multiscale_mode="range",
            ),
            Normalize=dict(),
            MultiScaleFlipAug=dict(
                img_scale=(1344, 800),
                flip=False,
                transforms=[
                    dict(type="Resize", keep_ratio=False),
                    dict(type="Normalize"),
                    dict(type="Pad", size_divisor=32),
                    dict(type="ImageToTensor", keys=["img"]),
                    dict(type="Collect", keys=["img"]),
                ],
            ),
        )
        self.configurer.configure_data(self.model_cfg, True, data_cfg)
        assert self.model_cfg.data
        assert self.model_cfg.data.train
        assert self.model_cfg.data.val

    @e2e_pytest_unit
    def test_configure_task(self, mocker):
        ssd_dir = os.path.join("src/otx/algorithms/detection/configs/detection", "mobilenetv2_ssd")
        ssd_cfg = MPAConfig.fromfile(os.path.join(ssd_dir, "model.py"))
        ssd_cfg.task_adapt = {"type": "mpa", "op": "REPLACE", "use_mpa_anchor": True}
        model_cfg = copy.deepcopy(ssd_cfg)
        self.configurer.configure_task(model_cfg, self.det_dataset, True)
        assert model_cfg.model.bbox_head.anchor_generator != ssd_cfg.model.bbox_head.anchor_generator

        model_cfg = copy.deepcopy(self.model_cfg)
        model_cfg.task_adapt = {"type": "mpa", "op": "REPLACE", "use_mpa_anchor": True}
        model_cfg.model.bbox_head.type = "ATSSHead"
        self.configurer.configure_task(model_cfg, self.det_dataset, True)

        model_cfg.model.bbox_head.type = "VFNetHead"
        self.configurer.configure_task(model_cfg, self.det_dataset, True)

        model_cfg.model.bbox_head.type = "YOLOXHead"
        model_cfg.data.train.type = "MultiImageMixDataset"
        self.configurer.configure_task(model_cfg, self.det_dataset, True)

        def mock_configure_classes(*args, **kwargs):
            return True

        mocker.patch.object(DetectionConfigurer, "configure_classes")
        self.configurer.model_classes = []
        self.configurer.data_classes = ["red", "green"]
        self.configurer.configure_classes = mock_configure_classes
        self.configurer.configure_task(model_cfg, self.det_dataset, True)

    @e2e_pytest_unit
    def test_configure_hook(self):
        model_cfg = copy.deepcopy(self.model_cfg)
        model_cfg.custom_hook_options = {"LazyEarlyStoppingHook": {"start": 5}, "LoggerReplaceHook": {"_delete_": True}}
        self.configurer.configure_hook(model_cfg)
        assert model_cfg.custom_hooks[0]["start"] == 5

    @e2e_pytest_unit
    def test_configure_samples_per_gpu(self):
        model_cfg = copy.deepcopy(self.model_cfg)
        model_cfg.data.train.otx_dataset = range(1)
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
    def test_configure_regularization(self):
        configure_cfg = copy.deepcopy(self.model_cfg)
        configure_cfg.model.l2sp_weight = 1.0
        self.configurer.configure_regularization(configure_cfg, True)
        assert "l2sp_ckpt" in configure_cfg.model
        assert configure_cfg.optimizer.weight_decay == 0.0

    @e2e_pytest_unit
    def test_get_data_cfg(self):
        config = copy.deepcopy(self.model_cfg)
        config.data.train.dataset = ConfigDict({"dataset": [1, 2, 3]})
        assert [1, 2, 3] == self.configurer.get_data_cfg(config, "train")


class TestIncrDetectionConfigurer:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.configurer = IncrDetectionConfigurer()
        self.model_cfg = MPAConfig.fromfile(os.path.join(DEFAULT_DET_TEMPLATE_DIR, "model.py"))
        self.data_cfg = MPAConfig.fromfile(os.path.join(DEFAULT_DET_TEMPLATE_DIR, "data_pipeline.py"))
        self.det_dataset, self.det_labels = generate_det_dataset(TaskType.DETECTION, 100)

    def test_configure_task(self, mocker):
        mocker.patch.object(DetectionConfigurer, "configure_task")
        self.model_cfg.task_adapt = {}
        self.configurer.task_adapt_type = "mpa"
        self.configurer.configure_task(self.model_cfg, self.det_dataset, True)
        assert self.model_cfg.custom_hooks[1].type == "TaskAdaptHook"
        assert self.model_cfg.custom_hooks[1].sampler_flag is False


class TestSemiSLDetectionConfigurer:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.configurer = SemiSLDetectionConfigurer()
        self.model_cfg = MPAConfig.fromfile(os.path.join(DEFAULT_DET_TEMPLATE_DIR, "model.py"))
        self.data_cfg = MPAConfig.fromfile(os.path.join(DEFAULT_DET_TEMPLATE_DIR, "data_pipeline.py"))
        self.det_dataset, self.det_labels = generate_det_dataset(TaskType.DETECTION, 100)

    def test_configure_data(self, mocker):
        mocker.patch.object(DetectionConfigurer, "configure_data")
        mocker.patch("mmdet.datasets.build_dataset", return_value=[])
        mocker.patch("otx.algorithms.detection.adapters.mmdet.configurer.build_dataloader", return_value=[])
        self.model_cfg.data.unlabeled = ConfigDict({"type": "OTXDataset", "otx_dataset": range(10)})
        self.model_cfg.model_task = "detection"
        self.model_cfg.distributed = False
        self.configurer.configure_data(self.model_cfg, True, self.data_cfg)

    def test_configure_task(self):
        self.model_cfg.task_adapt = {"type": "mpa", "op": "REPLACE", "use_mpa_anchor": True}
        self.configurer.configure_task(self.model_cfg, self.det_dataset, True)

        self.model_cfg.task_adapt = {"type": "not_mpa", "op": "REPLACE", "use_mpa_anchor": True}
        self.configurer.configure_task(self.model_cfg, self.det_dataset, True)
