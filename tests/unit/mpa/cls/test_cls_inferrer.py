import pytest

from otx.mpa.cls.inferrer import ClsInferrer
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.algorithms.classification.test_helper import setup_mpa_task_parameters


class TestOTXClsInferrer:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.model_cfg, self.data_cfg, recipie_cfg = setup_mpa_task_parameters(
            task_type="incremental", create_test=True
        )
        self.inferrer = ClsInferrer(name="", mode="train", config=recipie_cfg, common_cfg=None, index=0)

    @e2e_pytest_unit
    def test_run(self, mocker):
        fake_output = {"classes": [1, 2], "eval_predictions": None, "feature_vectors": None}
        self.inferrer.extract_prob = False
        mock_infer = mocker.patch.object(ClsInferrer, "infer", return_value=fake_output)

        returned_value = self.inferrer.run(self.model_cfg, "", self.data_cfg)
        mock_infer.assert_called_once()
        assert returned_value == {"outputs": fake_output}

    @e2e_pytest_unit
    def test_infer(self, mocker):
        cfg = self.inferrer.configure(self.model_cfg, "", self.data_cfg, training=False)
        mocker.patch.object(ClsInferrer, "configure_samples_per_gpu")
        mocker.patch.object(ClsInferrer, "configure_compat_cfg")
        mock_infer_callback = mocker.patch.object(ClsInferrer, "set_inference_progress_callback")
        mocker.patch("otx.mpa.cls.inferrer.build_data_parallel")
        mock_build_model = mocker.patch.object(ClsInferrer, "build_model")

        returned_value = self.inferrer.infer(cfg)
        mock_infer_callback.assert_called_once()
        mock_build_model.assert_called_once()

        assert "saliency_maps" in returned_value
        assert "eval_predictions" in returned_value
        assert "feature_vectors" in returned_value
        assert len(returned_value["eval_predictions"]) >= 0
