# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from typing import Callable
from unittest.mock import MagicMock

import numpy as np
import onnx
import pytest
import torch

from getitune.backend.lightning.exporter.native import LightningModelExporter
from getitune.backend.lightning.models.base import DataInputParams
from getitune.types.export import TaskLevelExportParameters
from getitune.types.precision import Precision


class TestLightningModelExporter:
    @pytest.fixture
    def fxt_exporter(self):
        # Create an instance of LightningModelExporter with default params
        return LightningModelExporter(
            task_level_export_parameters=MagicMock(spec=TaskLevelExportParameters),
            data_input_params=DataInputParams((224, 224), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        )

    @pytest.fixture
    def fxt_dummy_model(self):
        # Define a simple dummy torch model for testing
        return torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
        )

    @pytest.fixture
    def fxt_dummy_input(self) -> Callable[[int], np.ndarray]:
        rng = np.random.default_rng(seed=42)

        def _generate_dummy_input(batch_size) -> np.ndarray:
            return rng.random((batch_size, 3, 224, 224)).astype(np.float32)

        return _generate_dummy_input

    def test_to_openvino_export(self, fxt_exporter, fxt_dummy_model, tmp_path):
        # Use tmp_path provided by pytest for temporary file creation
        output_dir = tmp_path / "model_export"
        output_dir.mkdir()

        # Call the to_openvino method
        exported_path = fxt_exporter.to_openvino(
            model=fxt_dummy_model,
            output_dir=output_dir,
            base_model_name="test_model",
            precision=Precision.FP32,
        )

        # Check that the exported files exist
        assert exported_path.exists()
        assert (output_dir / "test_model.xml").exists()
        assert (output_dir / "test_model.bin").exists()

        fxt_exporter.via_onnx = True
        exported_path = fxt_exporter.to_openvino(
            model=fxt_dummy_model,
            output_dir=output_dir,
            base_model_name="test_model",
            precision=Precision.FP32,
        )

        assert exported_path.exists()
        assert (output_dir / "test_model.xml").exists()
        assert (output_dir / "test_model.bin").exists()

    def test_to_onnx_export(self, fxt_exporter, fxt_dummy_model, tmp_path):
        # Use tmp_path provided by pytest for temporary file creation
        output_dir = tmp_path / "onnx_export"
        output_dir.mkdir()

        # Call the to_onnx method
        exported_path = fxt_exporter.to_onnx(
            model=fxt_dummy_model,
            output_dir=output_dir,
            base_model_name="test_onnx_model",
            precision=Precision.FP32,
        )

        # Check that the exported ONNX file exists
        assert exported_path.exists()
        assert (output_dir / "test_onnx_model.onnx").exists()

        # Load the model to verify it's a valid ONNX file
        onnx_model = onnx.load(str(exported_path))
        onnx.checker.check_model(onnx_model)

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_to_openvino_export_dynamic_batch(
        self, batch_size, fxt_exporter, fxt_dummy_model, fxt_dummy_input, tmp_path
    ):
        """Exported OpenVINO model should accept batch sizes other than 1."""
        import openvino as ov

        output_dir = tmp_path / "model_export_dynamic"
        output_dir.mkdir()

        exported_path = fxt_exporter.to_openvino(
            model=fxt_dummy_model,
            output_dir=output_dir,
            base_model_name="test_model_dynamic",
            precision=Precision.FP32,
        )

        core = ov.Core()
        ov_model = core.read_model(exported_path)
        compiled_model = core.compile_model(ov_model, "CPU")

        output = compiled_model(fxt_dummy_input(batch_size))

        # Sanity check: output batch dim matches input batch dim
        result = next(iter(output.values()))
        assert result.shape[0] == batch_size

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_to_openvino_export_dynamic_batch_via_onnx(
        self, batch_size, fxt_exporter, fxt_dummy_model, fxt_dummy_input, tmp_path
    ):
        """Same dynamic-batch check for the via_onnx=True export path."""
        import openvino as ov

        output_dir = tmp_path / "model_export_dynamic_onnx"
        output_dir.mkdir()

        fxt_exporter.via_onnx = True
        exported_path = fxt_exporter.to_openvino(
            model=fxt_dummy_model,
            output_dir=output_dir,
            base_model_name="test_model_dynamic_onnx",
            precision=Precision.FP32,
        )

        core = ov.Core()
        ov_model = core.read_model(exported_path)
        compiled_model = core.compile_model(ov_model, "CPU")

        output = compiled_model(fxt_dummy_input(batch_size))
        result = next(iter(output.values()))
        assert result.shape[0] == batch_size

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_to_onnx_export_dynamic_batch(self, batch_size, fxt_exporter, fxt_dummy_model, fxt_dummy_input, tmp_path):
        """Exported ONNX model's input/output batch axis should be dynamic (symbolic), not fixed to 1."""
        output_dir = tmp_path / "onnx_export_dynamic"
        output_dir.mkdir()

        fxt_exporter.onnx_export_configuration = {
            "dynamic_shapes": ({0: torch.export.Dim("batch")},),
        }

        exported_path = fxt_exporter.to_onnx(
            model=fxt_dummy_model,
            output_dir=output_dir,
            base_model_name="test_onnx_model_dynamic",
            precision=Precision.FP32,
        )

        onnx_model = onnx.load(str(exported_path))

        def batch_dim_is_dynamic(value_info) -> bool:
            dim = value_info.type.tensor_type.shape.dim[0]
            # Symbolic dynamic dims have dim_param set, while dim_value stays at the default 0
            return (dim.dim_param != "") and (dim.dim_value == 0)

        assert batch_dim_is_dynamic(onnx_model.graph.input[0])
        for output in onnx_model.graph.output:
            assert batch_dim_is_dynamic(output)

        # Also functionally verify via onnxruntime with batch=8
        import onnxruntime as ort

        session = ort.InferenceSession(str(exported_path))
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: fxt_dummy_input(batch_size)})
        assert outputs[0].shape[0] == batch_size  # pyrefly: ignore[missing-attribute]
