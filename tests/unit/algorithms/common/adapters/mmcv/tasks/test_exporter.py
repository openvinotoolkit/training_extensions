from otx.api.usecases.tasks.interfaces.export_interface import ExportType
import mmcv
import pytest

from otx.algorithms.common.adapters.mmcv.tasks.exporter import Exporter
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class TestExporter:
    @pytest.fixture(autouse=True)
    def setup(self, mocker):
        def mock_init_logger():
            pass

        def mock_configure(model_cfg, model_ckpt, data_cfg, training=False, **kwargs):
            return mmcv.ConfigDict()

        self.exporter = Exporter()
        self.exporter._init_logger = mock_init_logger
        self.fake_config = mmcv.ConfigDict(work_dir="/path/work_dir", data=dict(test=dict(dataset=mocker.MagicMock())))
        mocker.patch("os.listdir")

    @e2e_pytest_unit
    def test_run_with_error_raise(self):
        return_value = self.exporter.run(self.fake_config)

        assert "outputs" in return_value
        assert return_value["outputs"] is None
        assert "msg" in return_value

    @e2e_pytest_unit
    def test_run_without_deploy_cfg(self, mocker):
        def mock_naive_export(output_dir, model_builder, precision, export_type, cfg, model_name="model"):
            pass

        self.exporter.naive_export = mock_naive_export
        return_value = self.exporter.run(self.fake_config)

        assert "outputs" in return_value
        assert return_value["outputs"]["bin"] == "/path/work_dir/model.bin"
        assert return_value["outputs"]["xml"] == "/path/work_dir/model.xml"
        assert "msg" in return_value
        assert return_value["msg"] == ""

    @e2e_pytest_unit
    def test_run_with_deploy_cfg(self, mocker):
        def mock_mmdeploy_export(
            output_dir, model_builder, precision, export_type, cfg, deploy_cfg, model_name="model"
        ):
            pass

        self.exporter.mmdeploy_export = mock_mmdeploy_export
        return_value = self.exporter.run(self.fake_config, deploy_cfg=mmcv.ConfigDict())

        assert "outputs" in return_value
        assert return_value["outputs"]["bin"] == "/path/work_dir/model.bin"
        assert return_value["outputs"]["xml"] == "/path/work_dir/model.xml"
        assert "msg" in return_value
        assert return_value["msg"] == ""

    @e2e_pytest_unit
    def test_mmdeploy_export(self, mocker):
        from otx.algorithms.common.adapters.mmdeploy.apis import MMdeployExporter

        mock_export_openvino = mocker.patch.object(MMdeployExporter, "export2backend")

        Exporter.mmdeploy_export(
            "",
            None,
            "FP16",
            ExportType.OPENVINO,
            dict(),
            mmcv.ConfigDict(backend_config=dict(mo_options=dict(flags=[]))),
        )

        mock_export_openvino.assert_called_once()
