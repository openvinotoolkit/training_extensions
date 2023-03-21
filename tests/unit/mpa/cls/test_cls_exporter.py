import pytest

from otx.algorithms.classification.adapters.mmcls.utils.builder import build_classifier
from otx.algorithms.common.adapters.mmdeploy.apis import NaiveExporter
from otx.mpa.cls.exporter import ClsExporter
from otx.mpa.exporter_mixin import ExporterMixin
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.algorithms.classification.test_helper import setup_mpa_task_parameters


class TestOTXClsExporter:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.model_cfg, self.data_cfg, recipie_cfg = setup_mpa_task_parameters(
            task_type="incremental", create_test=True
        )
        self.exporter = ClsExporter(name="", mode="train", config=recipie_cfg, common_cfg=None, index=0)

    @e2e_pytest_unit
    def test_run(self, mocker):
        args = {"precision": "FP32", "model_builder": build_classifier}
        mocker.patch.object(ExporterMixin, "run", return_value=True)
        returned_value = self.exporter.run(self.model_cfg, "", self.data_cfg, **args)

        assert returned_value is True

    @e2e_pytest_unit
    def test_naive_export(self, mocker):
        mock_export_ov = mocker.patch.object(NaiveExporter, "export2openvino")
        self.exporter.naive_export("", build_classifier, "FP32", self.data_cfg)

        mock_export_ov.assert_called_once()
