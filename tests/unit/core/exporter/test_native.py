import onnx
import pytest
import torch
from otx.core.exporter.native import OTXNativeModelExporter
from otx.core.types.export import TaskLevelExportParameters
from otx.core.types.precision import OTXPrecisionType


class TestOTXNativeModelExporter:
    @pytest.fixture()
    def exporter(self, mocker):
        # Create an instance of OTXNativeModelExporter with default params
        return OTXNativeModelExporter(
            task_level_export_parameters=mocker.MagicMock(TaskLevelExportParameters),
            input_size=(3, 224, 224),
        )

    @pytest.fixture()
    def dummy_model(self):
        # Define a simple dummy torch model for testing
        return torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
        )

    def test_to_openvino_export(self, exporter, dummy_model, tmp_path):
        # Use tmp_path provided by pytest for temporary file creation
        output_dir = tmp_path / "model_export"
        output_dir.mkdir()

        # Call the to_openvino method
        exported_path = exporter.to_openvino(
            model=dummy_model,
            output_dir=output_dir,
            base_model_name="test_model",
            precision=OTXPrecisionType.FP32,
        )

        # Check that the exported files exist
        assert exported_path.exists()
        assert (output_dir / "test_model.xml").exists()
        assert (output_dir / "test_model.bin").exists()

        exporter.via_onnx = True
        exported_path = exporter.to_openvino(
            model=dummy_model,
            output_dir=output_dir,
            base_model_name="test_model",
            precision=OTXPrecisionType.FP32,
        )

        assert exported_path.exists()
        assert (output_dir / "test_model.xml").exists()
        assert (output_dir / "test_model.bin").exists()

    def test_to_onnx_export(self, exporter, dummy_model, tmp_path):
        # Use tmp_path provided by pytest for temporary file creation
        output_dir = tmp_path / "onnx_export"
        output_dir.mkdir()

        # Call the to_onnx method
        exported_path = exporter.to_onnx(
            model=dummy_model,
            output_dir=output_dir,
            base_model_name="test_onnx_model",
            precision=OTXPrecisionType.FP32,
        )

        # Check that the exported ONNX file exists
        assert exported_path.exists()
        assert (output_dir / "test_onnx_model.onnx").exists()

        # Load the model to verify it's a valid ONNX file
        onnx_model = onnx.load(str(exported_path))
        onnx.checker.check_model(onnx_model)
