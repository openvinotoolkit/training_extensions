import pytest

from otx.mpa.cls.trainer import ClsTrainer
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.algorithms.classification.test_helper import setup_mpa_task_parameters


class TestOTXClsTrainer:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.model_cfg, self.data_cfg, recipie_cfg = setup_mpa_task_parameters(task_type="incremental")
        self.trainer = ClsTrainer(name="", mode="train", config=recipie_cfg, common_cfg=None, index=0)

    @e2e_pytest_unit
    def test_run(self, mocker):
        mocker.patch.object(ClsTrainer, "configure_samples_per_gpu")
        mocker.patch.object(ClsTrainer, "configure_fp16_optimizer")
        mocker.patch.object(ClsTrainer, "configure_compat_cfg")
        mock_train_classifier = mocker.patch("otx.mpa.cls.trainer.train_model")
        mocker.patch.object(ClsTrainer, "build_model")

        self.trainer.run(self.model_cfg, "", self.data_cfg)
        mock_train_classifier.assert_called_once()

    @e2e_pytest_unit
    def test_run_with_distributed(self, mocker):
        self.trainer._distributed = True
        mocker.patch.object(ClsTrainer, "configure_samples_per_gpu")
        mocker.patch.object(ClsTrainer, "configure_fp16_optimizer")
        mocker.patch.object(ClsTrainer, "configure_compat_cfg")
        spy_cfg_dist = mocker.spy(ClsTrainer, "_modify_cfg_for_distributed")
        mocker.patch.object(ClsTrainer, "build_model")
        mock_train_classifier = mocker.patch("otx.mpa.cls.trainer.train_model")

        self.trainer.run(self.model_cfg, "", self.data_cfg)
        spy_cfg_dist.assert_called_once()
        mock_train_classifier.assert_called_once()
