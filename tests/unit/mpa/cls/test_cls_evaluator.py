import pytest

from otx.algorithms.classification.adapters.mmcls.utils.builder import build_classifier
from otx.mpa.cls.evaluator import ClsEvaluator
from otx.mpa.cls.inferrer import ClsInferrer
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.algorithms.classification.test_helper import setup_mpa_task_parameters


class TestOTXClsEvaluator:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.model_cfg, self.data_cfg, recipie_cfg = setup_mpa_task_parameters(
            task_type="incremental", create_test=True
        )
        self.evaluator = ClsEvaluator(name="", mode="train", config=recipie_cfg, common_cfg=None, index=0)

    @e2e_pytest_unit
    def test_run(self, mocker):
        args = {"precision": "FP32", "model_builder": build_classifier}
        mocker.patch.object(ClsInferrer, "infer", return_value="")
        self.evaluator.dataset = mocker.MagicMock()
        returned_value = self.evaluator.run(self.model_cfg, "", self.data_cfg, **args)

        assert returned_value is not None
