# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import numpy as np
import onnx
import pytest
from onnx import ModelProto, TensorProto, helper, numpy_helper

from getitune.backend.lightning.exporter.base import (
    ExportFormat,
    ModelExporter,
    Precision,
    _convert_onnx_to_float16,
)
from getitune.backend.lightning.models.base import DataInputParams
from getitune.config.data import IntensityConfig
from getitune.types.export import TaskLevelExportParameters


class MockModelExporter(ModelExporter):
    def to_openvino(self, model, output_dir, base_model_name, precision):
        return output_dir / f"{base_model_name}.xml"

    def to_onnx(self, model, output_dir, base_model_name, precision):
        return output_dir / f"{base_model_name}.onnx"


@pytest.fixture
def mock_model():
    return MagicMock()


@pytest.fixture
def exporter():
    return MockModelExporter(
        task_level_export_parameters=MagicMock(TaskLevelExportParameters),
        data_input_params=DataInputParams((224, 224), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
    )


class TestLightningModelExporter:
    def test_to_openvino(self, mock_model, exporter, tmp_path):
        output_dir = tmp_path
        base_model_name = "test_model"
        precision = Precision.FP32
        result = exporter.export(mock_model, output_dir, base_model_name, ExportFormat.OPENVINO, precision)
        assert result == output_dir / f"{base_model_name}.xml"

    def test_to_onnx(self, mock_model, exporter, tmp_path):
        output_dir = tmp_path
        base_model_name = "test_model"
        precision = Precision.FP32
        result = exporter.export(mock_model, output_dir, base_model_name, ExportFormat.ONNX, precision)
        assert result == output_dir / f"{base_model_name}.onnx"

    def test_export_unsupported_format_raises(self, exporter, mock_model, tmp_path):
        export_format = "unsupported_format"
        with pytest.raises(ValueError, match=f"Unsupported export format: {export_format}"):
            exporter.export(mock_model, tmp_path, export_format=export_format)

    def test_postprocess_openvino_model(self, mock_model, exporter):
        # test output names do not match exporter parameters
        exporter.output_names = ["output1"]
        with pytest.raises(RuntimeError):
            exporter._postprocess_openvino_model(mock_model)
        # test output names match exporter parameters
        exporter.output_names = ["output1", "output2"]
        mock_model.outputs = []
        for output_name in exporter.output_names:
            output = MagicMock()
            output.get_names.return_value = output_name
            mock_model.outputs.append(output)
        processed_model = exporter._postprocess_openvino_model(mock_model)
        # Verify the processed model is returned and the names are set correctly
        assert processed_model is mock_model
        for output, name in zip(processed_model.outputs, exporter.output_names):
            output.tensor.set_names.assert_called_once_with({name})

    def test_embed_metadata_true_precision_fp16(self, exporter):
        onnx_model = ModelProto()
        exporter._embed_onnx_metadata = MagicMock(return_value=onnx_model)
        convert_mock = MagicMock(return_value=onnx_model)
        with pytest.MonkeyPatch.context() as m:
            m.setattr(
                "getitune.backend.lightning.exporter.base.convert_float_to_float16",
                convert_mock,
                raising=False,
            )
            # Need to re-import to pick up the monkeypatch inside the function scope
            from onnxruntime.transformers import float16 as ort_float16

            m.setattr(ort_float16, "convert_float_to_float16", convert_mock)
            result = exporter._postprocess_onnx_model(onnx_model, embed_metadata=True, precision=Precision.FP16)
            exporter._embed_onnx_metadata.assert_called_once()
            convert_mock.assert_called_once()
            assert result is onnx_model


class TestConvertOnnxToFloat16:
    """Tests for FP16 conversion using onnxruntime's float16 converter."""

    @staticmethod
    def _make_simple_fp32_model() -> ModelProto:
        """Create a minimal FP32 ONNX model (MatMul) for conversion testing."""
        x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])
        y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])
        w = numpy_helper.from_array(np.ones((4, 4), dtype=np.float32), name="W")
        matmul = helper.make_node("MatMul", ["X", "W"], ["Y"])
        graph = helper.make_graph([matmul], "test_graph", [x], [y], initializer=[w])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 7
        return model

    @staticmethod
    def _make_multi_consumer_cast_model() -> ModelProto:
        """Create an ONNX graph where a single node fans out to two consumers.

        This replicates the topology that triggered the onnxconverter_common bug
        (issue #5439): a node whose output feeds multiple downstream nodes.
        ORT's converter handles this correctly without workarounds.
        """
        x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])
        out1 = helper.make_tensor_value_info("out1", TensorProto.FLOAT, [1, 4])
        out2 = helper.make_tensor_value_info("out2", TensorProto.FLOAT, [1, 4])

        w1 = numpy_helper.from_array(np.ones((4, 4), dtype=np.float32), name="W1")
        w2 = numpy_helper.from_array(np.ones((4, 4), dtype=np.float32), name="W2")

        # Relu fans out to two MatMul consumers (common in ROI-style subgraphs)
        relu = helper.make_node("Relu", ["X"], ["relu_out"], name="relu")
        mm1 = helper.make_node("MatMul", ["relu_out", "W1"], ["out1"], name="mm1")
        mm2 = helper.make_node("MatMul", ["relu_out", "W2"], ["out2"], name="mm2")

        graph = helper.make_graph(
            [relu, mm1, mm2],
            "multi_consumer_graph",
            [x],
            [out1, out2],
            initializer=[w1, w2],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 7
        return model

    def test_simple_model_converts_to_fp16(self):
        model = self._make_simple_fp32_model()
        result = _convert_onnx_to_float16(model)
        # Verify weights are converted to FP16
        fp16_inits = [i for i in result.graph.initializer if i.data_type == TensorProto.FLOAT16]
        assert len(fp16_inits) > 0

    def test_io_types_preserved_as_fp32(self):
        """Model I/O should remain FP32 (keep_io_types=True)."""
        model = self._make_simple_fp32_model()
        result = _convert_onnx_to_float16(model)
        for inp in result.graph.input:
            if inp.type.tensor_type.elem_type != 0:  # skip non-typed (initializers)
                assert inp.type.tensor_type.elem_type == TensorProto.FLOAT
        for out in result.graph.output:
            assert out.type.tensor_type.elem_type == TensorProto.FLOAT

    def test_multi_consumer_model_does_not_crash(self):
        """Regression test for issue #5439: multi-consumer nodes must not crash."""
        model = self._make_multi_consumer_cast_model()
        result = _convert_onnx_to_float16(model)
        assert result is not None
        onnx.checker.check_model(result)

    def test_converted_model_is_valid_onnx(self):
        """Converted model must pass ONNX checker."""
        model = self._make_simple_fp32_model()
        result = _convert_onnx_to_float16(model)
        onnx.checker.check_model(result)

    def test_extend_model_metadata_no_intensity(self, exporter):
        """Without intensity_config, metadata should contain only standard preprocessing keys."""
        metadata = exporter._extend_model_metadata({})
        assert ("model_info", "mean_values") in metadata
        assert ("model_info", "scale_values") in metadata
        assert ("model_info", "resize_type") in metadata
        assert ("model_info", "pad_value") in metadata
        assert ("model_info", "reverse_input_channels") in metadata
        # No intensity keys
        assert ("model_info", "input_dtype") not in metadata
        assert ("model_info", "intensity_mode") not in metadata

    def test_extend_model_metadata_with_scale_to_unit_intensity(self):
        """With scale_to_unit IntensityConfig, intensity metadata should be embedded."""
        intensity_cfg = IntensityConfig(
            storage_dtype="uint16",
            mode="scale_to_unit",
            max_value=65535.0,
        )
        exporter = MockModelExporter(
            task_level_export_parameters=MagicMock(TaskLevelExportParameters),
            data_input_params=DataInputParams(
                (224, 224),
                (0.0, 0.0, 0.0),
                (1.0, 1.0, 1.0),
                intensity_config=intensity_cfg,
            ),
        )
        metadata = exporter._extend_model_metadata({})

        assert metadata[("model_info", "input_dtype")] == "u16"
        assert metadata[("model_info", "intensity_mode")] == "scale_to_unit"
        assert metadata[("model_info", "intensity_max_value")] == "65535.0"
        assert metadata[("model_info", "intensity_scale_factor")] == "1.0"
        assert metadata[("model_info", "intensity_min_value")] == "0.0"
        assert metadata[("model_info", "intensity_percentile_low")] == "1.0"
        assert metadata[("model_info", "intensity_percentile_high")] == "99.0"
        # No repeat_channels when it's 0
        assert ("model_info", "intensity_repeat_channels") not in metadata
        # window_center and window_width should not be present when None
        assert ("model_info", "intensity_window_center") not in metadata
        assert ("model_info", "intensity_window_width") not in metadata

    def test_extend_model_metadata_with_window_intensity(self):
        """With window IntensityConfig, window parameters should be embedded."""
        intensity_cfg = IntensityConfig(
            storage_dtype="uint16",
            mode="window",
            window_center=40.0,
            window_width=400.0,
            repeat_channels=3,
        )
        exporter = MockModelExporter(
            task_level_export_parameters=MagicMock(TaskLevelExportParameters),
            data_input_params=DataInputParams(
                (512, 512),
                (0.0, 0.0, 0.0),
                (1.0, 1.0, 1.0),
                intensity_config=intensity_cfg,
            ),
        )
        metadata = exporter._extend_model_metadata({})

        assert metadata[("model_info", "input_dtype")] == "u16"
        assert metadata[("model_info", "intensity_mode")] == "window"
        assert metadata[("model_info", "intensity_window_center")] == "40.0"
        assert metadata[("model_info", "intensity_window_width")] == "400.0"
        assert metadata[("model_info", "intensity_repeat_channels")] == "True"

    def test_extend_model_metadata_with_range_scale_intensity(self):
        """With range_scale IntensityConfig (thermal), all range params should be embedded."""
        intensity_cfg = IntensityConfig(
            storage_dtype="uint16",
            mode="range_scale",
            scale_factor=0.4,
            min_value=295.15,
            max_value=360.15,
            repeat_channels=3,
        )
        exporter = MockModelExporter(
            task_level_export_parameters=MagicMock(TaskLevelExportParameters),
            data_input_params=DataInputParams(
                (640, 640),
                (0.0, 0.0, 0.0),
                (1.0, 1.0, 1.0),
                intensity_config=intensity_cfg,
            ),
        )
        metadata = exporter._extend_model_metadata({})

        assert metadata[("model_info", "input_dtype")] == "u16"
        assert metadata[("model_info", "intensity_mode")] == "range_scale"
        assert metadata[("model_info", "intensity_scale_factor")] == "0.4"
        assert metadata[("model_info", "intensity_min_value")] == "295.15"
        assert metadata[("model_info", "intensity_max_value")] == "360.15"
        assert metadata[("model_info", "intensity_repeat_channels")] == "True"

    def test_extend_model_metadata_with_percentile_intensity(self):
        """With percentile IntensityConfig, percentile params should be embedded."""
        intensity_cfg = IntensityConfig(
            storage_dtype="uint16",
            mode="percentile",
            percentile_low=2.0,
            percentile_high=98.0,
        )
        exporter = MockModelExporter(
            task_level_export_parameters=MagicMock(TaskLevelExportParameters),
            data_input_params=DataInputParams(
                (224, 224),
                (0.0, 0.0, 0.0),
                (1.0, 1.0, 1.0),
                intensity_config=intensity_cfg,
            ),
        )
        metadata = exporter._extend_model_metadata({})

        assert metadata[("model_info", "input_dtype")] == "u16"
        assert metadata[("model_info", "intensity_mode")] == "percentile"
        assert metadata[("model_info", "intensity_percentile_low")] == "2.0"
        assert metadata[("model_info", "intensity_percentile_high")] == "98.0"

    def test_extend_model_metadata_uint8_default_intensity(self):
        """Default IntensityConfig (uint8/scale_to_unit) should map to u8."""
        intensity_cfg = IntensityConfig()  # defaults: storage_dtype="uint8", mode="scale_to_unit"
        exporter = MockModelExporter(
            task_level_export_parameters=MagicMock(TaskLevelExportParameters),
            data_input_params=DataInputParams(
                (224, 224),
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
                intensity_config=intensity_cfg,
            ),
        )
        metadata = exporter._extend_model_metadata({})

        assert metadata[("model_info", "input_dtype")] == "u8"
        assert metadata[("model_info", "intensity_mode")] == "scale_to_unit"
        # max_value is None by default, so should NOT be written
        assert ("model_info", "intensity_max_value") not in metadata

    def test_extend_model_metadata_model_metadata_priority(self):
        """Model's original metadata should take priority over exporter's extra metadata."""
        intensity_cfg = IntensityConfig(storage_dtype="uint16", mode="scale_to_unit", max_value=65535.0)
        exporter = MockModelExporter(
            task_level_export_parameters=MagicMock(TaskLevelExportParameters),
            data_input_params=DataInputParams(
                (224, 224),
                (0.0, 0.0, 0.0),
                (1.0, 1.0, 1.0),
                intensity_config=intensity_cfg,
            ),
        )
        # Model metadata overrides
        model_metadata = {("model_info", "intensity_mode"): "custom_mode"}
        metadata = exporter._extend_model_metadata(model_metadata)

        # Model's value should take priority
        assert metadata[("model_info", "intensity_mode")] == "custom_mode"
        # Other intensity keys should still be present
        assert metadata[("model_info", "input_dtype")] == "u16"

    def test_extend_model_metadata_float32_dtype(self):
        """float32 storage_dtype should map to f32."""
        intensity_cfg = IntensityConfig(storage_dtype="float32", mode="scale_to_unit", max_value=1.0)
        exporter = MockModelExporter(
            task_level_export_parameters=MagicMock(TaskLevelExportParameters),
            data_input_params=DataInputParams(
                (224, 224),
                (0.0, 0.0, 0.0),
                (1.0, 1.0, 1.0),
                intensity_config=intensity_cfg,
            ),
        )
        metadata = exporter._extend_model_metadata({})
        assert metadata[("model_info", "input_dtype")] == "f32"

    def test_extend_model_metadata_unsupported_dtype_raises(self):
        """Unsupported storage_dtype must raise ValueError rather than silently embedding a raw value."""
        intensity_cfg = IntensityConfig(storage_dtype="bfloat16", mode="scale_to_unit", max_value=1.0)
        exporter = MockModelExporter(
            task_level_export_parameters=MagicMock(TaskLevelExportParameters),
            data_input_params=DataInputParams(
                (224, 224),
                (0.0, 0.0, 0.0),
                (1.0, 1.0, 1.0),
                intensity_config=intensity_cfg,
            ),
        )
        with pytest.raises(ValueError, match="Unsupported intensity storage_dtype 'bfloat16'"):
            exporter._extend_model_metadata({})


class TestDataInputParams:
    def test_as_dict_without_intensity(self):
        """as_dict without intensity_config should return only input_size, mean, std."""
        params = DataInputParams((224, 224), (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        result = params.as_dict()
        assert result == {
            "input_size": (224, 224),
            "mean": (0.485, 0.456, 0.406),
            "std": (0.229, 0.224, 0.225),
        }
        assert "intensity_config" not in result

    def test_as_dict_with_intensity(self):
        """as_dict with intensity_config should include intensity_config dict."""
        intensity_cfg = IntensityConfig(
            storage_dtype="uint16",
            mode="window",
            window_center=40.0,
            window_width=400.0,
            repeat_channels=3,
        )
        params = DataInputParams(
            (512, 512),
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
            intensity_config=intensity_cfg,
        )
        result = params.as_dict()
        assert "intensity_config" in result
        ic = result["intensity_config"]
        assert ic["storage_dtype"] == "uint16"
        assert ic["mode"] == "window"
        assert ic["window_center"] == 40.0
        assert ic["window_width"] == 400.0
        assert ic["repeat_channels"] == 3

    def test_default_intensity_config_is_none(self):
        """Default intensity_config should be None."""
        params = DataInputParams((224, 224), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        assert params.intensity_config is None

    def test_backward_compat_positional_args(self):
        """Existing code using 3 positional args should still work."""
        params = DataInputParams((224, 224), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        assert params.input_size == (224, 224)
        assert params.mean == (0.0, 0.0, 0.0)
        assert params.std == (1.0, 1.0, 1.0)
        assert params.intensity_config is None
