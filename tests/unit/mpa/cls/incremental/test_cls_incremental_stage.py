import pytest

from otx.mpa.cls.incremental.stage import IncrClsStage
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.algorithms.classification.test_helper import setup_mpa_task_parameters


class TestOTXIncrClsStage:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.model_cfg, self.data_cfg, recipie_cfg = setup_mpa_task_parameters(
            task_type="incremental", create_val=True, create_test=True
        )
        self.stage = IncrClsStage(name="", mode="train", config=recipie_cfg, common_cfg=None, index=0)

    @pytest.mark.parametrize("mode", ["MERGE", "REPLACE"])
    @e2e_pytest_unit
    def test_configure_classes(self, mode, mocker):

        self.stage.cfg.merge_from_dict(self.data_cfg)
        self.stage.cfg.task_adapt.op = mode
        origin_model_classes = ["label_0", "label_3", "label_n"]
        mocker.patch("otx.mpa.cls.incremental.stage.IncrClsStage.get_model_classes", return_value=origin_model_classes)
        self.stage.data_classes = self.data_cfg.data.train.data_classes  # ["label_0", "label_1"]
        self.stage.configure_classes(self.stage.cfg)
        merge_target = ["label_0", "label_1", "label_3", "label_n"]

        if mode == "REPLACE":
            assert self.stage.cfg.model.head.num_classes == len(self.stage.data_classes)
        else:
            assert self.stage.cfg.model.head.num_classes == len(merge_target)

    @e2e_pytest_unit
    def test_configure_task_modules(self, monkeypatch, mocker):
        mock_update_hook = mocker.patch("otx.mpa.cls.incremental.stage.update_or_add_custom_hook")
        # some dummy classes
        self.stage.data_classes = [0, 1]
        self.stage.model_classes = [0, 1]
        self.stage.org_model_classes = [0, 1]
        self.stage.configure_task_modules(self.stage.cfg)

        mock_update_hook.assert_called_once()
