import os
import tempfile

import pytest

from otx.algorithms.common.adapters.mmcv.utils.config_utils import MPAConfig
from otx.mpa.det.trainer import DetectionTrainer
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.algorithms.detection.test_helpers import (
    DEFAULT_DET_RECIPE_CONFIG_PATH,
    DEFAULT_DET_TEMPLATE_DIR,
    create_dummy_coco_json,
)


class TestDetectionTrainer:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        cfg = MPAConfig.fromfile(DEFAULT_DET_RECIPE_CONFIG_PATH)
        self.trainer = DetectionTrainer(name="", mode="train", config=cfg, common_cfg=None, index=0)
        self.model_cfg = MPAConfig.fromfile(os.path.join(DEFAULT_DET_TEMPLATE_DIR, "model.py"))
        self.data_cfg = MPAConfig.fromfile(os.path.join(DEFAULT_DET_TEMPLATE_DIR, "data_pipeline.py"))

    @e2e_pytest_unit
    def test_run(self, mocker):
        mocker.patch.object(DetectionTrainer, "configure_samples_per_gpu")
        mocker.patch.object(DetectionTrainer, "configure_fp16_optimizer")
        mocker.patch.object(DetectionTrainer, "configure_compat_cfg")
        mock_train_detector = mocker.patch("otx.mpa.det.trainer.train_detector")
        with tempfile.TemporaryDirectory() as tmp_dir:
            fake_json_path = os.path.join(tmp_dir, "fake_data.json")
            create_dummy_coco_json(fake_json_path)
            self.data_cfg.data.train.ann_file = fake_json_path
            self.data_cfg.data.train.data_classes = ["red", "green"]
            self.data_cfg.data.val.ann_file = fake_json_path
            self.data_cfg.data.test.ann_file = fake_json_path
            self.data_cfg.data.val_dataloader = dict()
            self.trainer.run(self.model_cfg, "", self.data_cfg)
            mock_train_detector.assert_called_once()

    @e2e_pytest_unit
    def test_run_with_distributed(self, mocker):
        self.trainer._distributed = True
        mocker.patch.object(DetectionTrainer, "configure_samples_per_gpu")
        mocker.patch.object(DetectionTrainer, "configure_fp16_optimizer")
        mocker.patch.object(DetectionTrainer, "configure_compat_cfg")
        spy_cfg_dist = mocker.spy(DetectionTrainer, "_modify_cfg_for_distributed")
        mock_train_detector = mocker.patch("otx.mpa.det.trainer.train_detector")

        with tempfile.TemporaryDirectory() as tmp_dir:
            fake_json_path = os.path.join(tmp_dir, "fake_data.json")
            create_dummy_coco_json(fake_json_path)
            self.data_cfg.data.train.ann_file = fake_json_path
            self.data_cfg.data.train.data_classes = ["red", "green"]
            self.data_cfg.data.val.ann_file = fake_json_path
            self.data_cfg.data.test.ann_file = fake_json_path
            self.data_cfg.data.val_dataloader = dict()
            self.trainer.run(self.model_cfg, "", self.data_cfg)
            spy_cfg_dist.assert_called_once()
            mock_train_detector.assert_called_once()
