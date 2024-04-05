from unittest.mock import MagicMock

import pytest
from onnx import ModelProto
from onnxconverter_common import float16
from otx.core.exporter.base import OTXExportFormatType, OTXModelExporter, OTXPrecisionType


class MockModelExporter(OTXModelExporter):
    def to_openvino(self, model, output_dir, base_model_name, precision):
        ov_file = output_dir / f"{base_model_name}.xml"
        (output_dir / f"{base_model_name}.bin").touch()
        ov_file.touch()
        return ov_file

    def to_onnx(self, model, output_dir, base_model_name, precision):
        onnx_file = output_dir / f"{base_model_name}.onnx"
        onnx_file.touch()
        return onnx_file


@pytest.fixture()
def mock_model():
    return MagicMock()


@pytest.fixture()
def exporter():
    return MockModelExporter(input_size=(224, 224), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))


@pytest.fixture()
def output_dir(tmp_path):
    return tmp_path


class TestOTXModelExporter:
    def test_to_openvino(self, mock_model, exporter, output_dir):
        base_model_name = "test_model"
        precision = OTXPrecisionType.FP32
        result = exporter.export(mock_model, output_dir, base_model_name, OTXExportFormatType.OPENVINO, precision)
        assert result == output_dir / f"{base_model_name}.xml"
        assert (output_dir / f"{base_model_name}.bin").exists()
        assert result.exists()

    def test_to_onnx(self, mock_model, exporter, output_dir):
        base_model_name = "test_model"
        precision = OTXPrecisionType.FP32
        result = exporter.export(mock_model, output_dir, base_model_name, OTXExportFormatType.ONNX, precision)
        assert result == output_dir / f"{base_model_name}.onnx"
        assert result.exists()

    def test_export_unsupported_format_raises(self, exporter, mock_model, output_dir):
        export_format = "unsupported_format"
        with pytest.raises(ValueError, match=f"Unsupported export format: {export_format}"):
            exporter.export(mock_model, output_dir, export_format=export_format)

    def test_to_exportable_code(self, mock_model, exporter, output_dir):
        base_model_name = "test_model"
        output_dir = output_dir / "exportable_code"
        precision = OTXPrecisionType.FP32

        result = exporter.to_exportable_code(mock_model, output_dir, base_model_name, precision)

        assert result == output_dir / "exportable_code.zip"

    def test_postprocess_openvino_model(self, mock_model, exporter):
        # test output names do not match exporter parameters
        exporter.output_names = ["output1"]
        with pytest.raises(RuntimeError):
            exporter._postprocess_openvino_model(mock_model)
        # test output names match exporter parameters
        exporter.output_names = ["output1", "output2"]
        mock_model.outputs = [MagicMock(), MagicMock()]
        processed_model = exporter._postprocess_openvino_model(mock_model)
        # Verify the processed model is returned and the names are set correctly
        assert processed_model is mock_model
        for output, name in zip(processed_model.outputs, exporter.output_names):
            output.tensor.set_names.assert_called_once_with({name})

    def test_embed_metadata_true_precision_fp16(self, exporter):
        onnx_model = ModelProto()
        exporter._embed_onnx_metadata = MagicMock(return_value=onnx_model)
        convert_float_to_float16_mock = MagicMock(return_value=onnx_model)
        with pytest.MonkeyPatch.context() as m:
            m.setattr(float16, "convert_float_to_float16", convert_float_to_float16_mock)
            result = exporter._postprocess_onnx_model(onnx_model, embed_metadata=True, precision=OTXPrecisionType.FP16)
            exporter._embed_onnx_metadata.assert_called_once()
            convert_float_to_float16_mock.assert_called_once_with(onnx_model)
            assert result is onnx_model