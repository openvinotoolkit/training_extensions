import copy

import pytest

from otx.algorithms.common.adapters.mmcv.hooks.recording_forward_hook import (
    ActivationMapHook,
)
from otx.mpa.cls.explainer import ClsExplainer
from otx.mpa.cls.stage import ClsStage
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.algorithms.classification.test_helper import (
    generate_cls_dataset,
    setup_mpa_task_parameters,
)


class TestOTXClsExplainer:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.model_cfg, self.data_cfg, recipie_cfg = setup_mpa_task_parameters(
            task_type="incremental", create_test=True
        )
        self.explainer = ClsExplainer(name="", mode="train", config=recipie_cfg, common_cfg=None, index=0)

    @e2e_pytest_unit
    def test_run(self, mocker):
        args = {"explainer": "activationmap"}
        mocker.patch.object(ClsExplainer, "explain", return_value="fake_output")
        returned_value = self.explainer.run(self.model_cfg, "", self.data_cfg, **args)

        assert returned_value == {"outputs": "fake_output"}

    @e2e_pytest_unit
    def test_explain(self, mocker):
        mocker.patch("otx.mpa.cls.explainer.build_data_parallel")
        mock_build_model = mocker.patch.object(ClsStage, "build_model")
        mocker.patch.object(ClsStage, "configure_samples_per_gpu")
        data_cfg = copy.deepcopy(self.data_cfg)
        dummy_dataset = generate_cls_dataset(number_of_images=1)
        data_cfg.data.test["otx_dataset"] = dummy_dataset
        data_cfg.data.test["labels"] = dummy_dataset.get_labels()
        self.explainer.cfg.merge_from_dict(self.model_cfg)
        self.explainer.cfg.merge_from_dict(data_cfg)
        self.explainer.explainer_hook = ActivationMapHook
        outputs = self.explainer.explain(self.explainer.cfg)

        mock_build_model.assert_called_once()
        assert outputs == {"saliency_maps": [], "eval_predictions": []}
