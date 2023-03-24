import copy
import os

import pytest

from otx.algorithms.common.adapters.mmcv.utils.config_utils import MPAConfig
from otx.mpa import Stage
from otx.mpa.det.stage import DetectionStage
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.algorithms.detection.test_helpers import (
    DEFAULT_DET_RECIPE_CONFIG_PATH,
    DEFAULT_DET_TEMPLATE_DIR,
)


class TestOTXDetectionStage:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        cfg = MPAConfig.fromfile(DEFAULT_DET_RECIPE_CONFIG_PATH)
        self.stage = DetectionStage(name="", mode="train", config=cfg, common_cfg=None, index=0)
        self.model_cfg = MPAConfig.fromfile(os.path.join(DEFAULT_DET_TEMPLATE_DIR, "model.py"))
        self.data_cfg = MPAConfig.fromfile(os.path.join(DEFAULT_DET_TEMPLATE_DIR, "data_pipeline.py"))

    @e2e_pytest_unit
    def test_configure(self, mocker):
        mock_cfg_model = mocker.patch.object(DetectionStage, "configure_model")
        mock_cfg_ckpt = mocker.patch.object(DetectionStage, "configure_ckpt")
        mock_cfg_regularization = mocker.patch.object(DetectionStage, "configure_regularization")
        mock_cfg_task = mocker.patch.object(DetectionStage, "configure_task")
        mock_cfg_hook = mocker.patch.object(DetectionStage, "configure_hook")

        fake_arg = {"pretrained": True, "foo": "bar"}
        returned_value = self.stage.configure(self.model_cfg, "", self.data_cfg, True, **fake_arg)
        mock_cfg_model.assert_called_once_with(self.stage.cfg, self.model_cfg, True, **fake_arg)
        mock_cfg_ckpt.assert_called_once_with(self.stage.cfg, "", fake_arg.get("pretrained", None))
        mock_cfg_regularization.assert_called_once_with(self.stage.cfg, True)
        mock_cfg_task.assert_called_once_with(self.stage.cfg, True, **fake_arg)
        mock_cfg_hook.assert_called_once_with(self.stage.cfg)
        assert returned_value == self.stage.cfg

    @e2e_pytest_unit
    def test_configure_model(self):
        fake_arg = {"ir_model_path": {"ir_weight_path": "", "ir_weight_init": ""}}
        self.stage.configure_model(self.stage.cfg, self.model_cfg, True, **fake_arg)
        assert self.stage.cfg.model_task

    @e2e_pytest_unit
    def test_configure_model_without_model(self):
        fake_arg = {"ir_model_path": {"ir_weight_path": "", "ir_weight_init": ""}}
        model_cfg = copy.deepcopy(self.model_cfg)
        model_cfg.pop("model")
        with pytest.raises(ValueError):
            self.stage.configure_model(self.stage.cfg, model_cfg, True, **fake_arg)

    @e2e_pytest_unit
    def test_configure_model_not_detection_task(self):
        fake_arg = {"ir_model_path": {"ir_weight_path": "", "ir_weight_init": ""}}
        stage_cfg = copy.deepcopy(self.stage.cfg)
        stage_cfg.model.task = "classification"
        with pytest.raises(ValueError):
            self.stage.configure_model(stage_cfg, self.model_cfg, True, **fake_arg)

    @e2e_pytest_unit
    def test_configure_data(self, mocker):
        mock_super_cfg_data = mocker.patch.object(Stage, "configure_data")
        self.stage.configure_data(self.stage.cfg, True, self.data_cfg, pretrained=None)
        mock_super_cfg_data.assert_called_once()
        assert self.stage.cfg.data
        assert self.stage.cfg.data.train
        assert self.stage.cfg.data.val

    @e2e_pytest_unit
    def test_configure_task(self, mocker):
        self.stage.data_classes = ["red", "green"]
        self.stage.model_classes = []
        mock_cfg_classes = mocker.patch.object(DetectionStage, "configure_classes")
        mocker.patch.object(DetectionStage, "configure_bbox_head")
        self.stage.configure_task(self.stage.cfg, True)
        mock_cfg_classes.assert_called_once()

    @e2e_pytest_unit
    def test_configure_regularization(self):
        stage_cfg = copy.deepcopy(self.stage.cfg)
        stage_cfg.model.l2sp_weight = 1.0
        self.stage.configure_regularization(stage_cfg, True)
        assert "l2sp_ckpt" in stage_cfg.model
        assert stage_cfg.optimizer.weight_decay == 0.0

    @e2e_pytest_unit
    def test_configure_hyperparams(self):
        stage_cfg = copy.deepcopy(self.stage.cfg)
        stage_cfg.hyperparams = dict()
        self.stage.configure_hyperparams(stage_cfg, True, hyperparams=dict(bs=2, lr=0.002))
        assert stage_cfg.data.samples_per_gpu == 2
        assert stage_cfg.optimizer.lr == 0.002

    @e2e_pytest_unit
    def test_configure_anchor(self):
        stage_cfg = copy.deepcopy(self.stage.cfg)
        stage_cfg.model.type = "SingleStageDetector"
        stage_cfg.merge_from_dict(
            dict(model=dict(bbox_head=dict(anchor_generator=dict(type="SSDAnchorGeneratorClustered"))))
        )
        self.stage.configure_anchor(stage_cfg, True)

    @e2e_pytest_unit
    def test_add_yolox_hooks(self):
        stage_cfg = copy.deepcopy(self.stage.cfg)
        self.stage.add_yolox_hooks(stage_cfg)
        custom_hooks = [hook.type for hook in stage_cfg.custom_hooks]
        assert "SyncNormHook" in custom_hooks
        assert "YOLOXModeSwitchHook" in custom_hooks

    @e2e_pytest_unit
    def test_configure_bbox_head(self):
        cfg = MPAConfig.fromfile(DEFAULT_DET_RECIPE_CONFIG_PATH)
        stage = DetectionStage(name="", mode="train", config=cfg, common_cfg=None, index=0)
        stage.cfg.merge_from_dict(dict(model=dict(bbox_head=dict(type="SSDHead"))))
        stage.model_classes = ["red", "green"]
        stage.configure_bbox_head(stage.cfg)
        if stage.cfg.get("ignore", False):
            assert stage.cfg.model.bbox_head.loss_cls.type == "CrossSigmoidFocalLoss"
