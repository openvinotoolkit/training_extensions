import os

import pytest

from otx.algorithms.common.adapters.mmcv.utils.config_utils import MPAConfig
from otx.mpa.seg.trainer import SegTrainer
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.algorithms.segmentation.test_helpers import (
    DEFAULT_RECIPE_CONFIG_PATH,
    DEFAULT_SEG_TEMPLATE_DIR,
)


class TestOTXSegTrainer:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        cfg = MPAConfig.fromfile(DEFAULT_RECIPE_CONFIG_PATH)
        self.trainer = SegTrainer(name="", mode="train", config=cfg, common_cfg=None, index=0)
        self.model_cfg = MPAConfig.fromfile(os.path.join(DEFAULT_SEG_TEMPLATE_DIR, "model.py"))
        self.data_cfg = MPAConfig.fromfile(os.path.join(DEFAULT_SEG_TEMPLATE_DIR, "data_pipeline.py"))

    @e2e_pytest_unit
    def test_run(self, mocker):
        mocker.patch.object(SegTrainer, "configure_samples_per_gpu")
        mocker.patch.object(SegTrainer, "configure_fp16_optimizer")
        mocker.patch.object(SegTrainer, "configure_compat_cfg")
        mock_train_segmentor = mocker.patch("otx.mpa.seg.trainer.train_segmentor")

        self.trainer.run(self.model_cfg, "", self.data_cfg)
        mock_train_segmentor.assert_called_once()

    @e2e_pytest_unit
    def test_run_with_distributed(self, mocker):
        self.trainer._distributed = True
        mocker.patch.object(SegTrainer, "configure_samples_per_gpu")
        mocker.patch.object(SegTrainer, "configure_fp16_optimizer")
        mocker.patch.object(SegTrainer, "configure_compat_cfg")
        spy_cfg_dist = mocker.spy(SegTrainer, "_modify_cfg_for_distributed")
        mock_train_segmentor = mocker.patch("otx.mpa.seg.trainer.train_segmentor")

        self.trainer.run(self.model_cfg, "", self.data_cfg)
        spy_cfg_dist.assert_called_once()
        mock_train_segmentor.assert_called_once()
