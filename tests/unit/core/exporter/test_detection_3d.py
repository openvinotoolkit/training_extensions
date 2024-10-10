# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Unit tests of visual prompting exporter."""

from unittest.mock import MagicMock

import pytest
import torch
from otx.core.exporter.detection_3d import OTXObjectDetection3DExporter
from otx.core.types.export import OTXExportFormatType


class TestOTXVisualPromptingModelExporter:
    @pytest.fixture()
    def otx_detection_3d_exporter(self) -> OTXObjectDetection3DExporter:
        return OTXObjectDetection3DExporter(
            task_level_export_parameters=MagicMock(),
            input_size=(10, 10),
        )

    def test_export_openvino(self, mocker, tmpdir, otx_detection_3d_exporter) -> None:
        """Test export for OPENVINO."""
        mocker_openvino_convert_model = mocker.patch("openvino.convert_model")
        mocker_postprocess_openvino_model = mocker.patch.object(
            otx_detection_3d_exporter,
            "_postprocess_openvino_model",
        )
        mocker_openvino_save_model = mocker.patch("openvino.save_model")
        mock_model = mocker.MagicMock()
        mock_model.parameters.return_value = iter([torch.rand(1, 3)])

        otx_detection_3d_exporter.export(
            model=mock_model,
            output_dir=tmpdir,
            export_format=OTXExportFormatType.OPENVINO,
        )

        mocker_openvino_convert_model.assert_called()
        mocker_postprocess_openvino_model.assert_called()
        mocker_openvino_save_model.assert_called()

        with pytest.raises(NotImplementedError):
            otx_detection_3d_exporter.export(
                model=mock_model,
                output_dir=tmpdir,
                export_format=OTXExportFormatType.OPENVINO,
                to_exportable_code=True,
            )

    def test_export_onnx(self, mocker, tmpdir, otx_detection_3d_exporter) -> None:
        """Test export for ONNX."""
        mocker_torch_onnx_export = mocker.patch("torch.onnx.export")
        mocker_onnx_load = mocker.patch("onnx.load")
        mocker_onnx_save = mocker.patch("onnx.save")
        mocker_postprocess_onnx_model = mocker.patch.object(
            otx_detection_3d_exporter,
            "_postprocess_onnx_model",
        )
        mock_model = mocker.MagicMock()

        otx_detection_3d_exporter.export(
            model=mock_model,
            output_dir=tmpdir,
            export_format=OTXExportFormatType.ONNX,
        )

        mocker_torch_onnx_export.assert_called()
        mocker_onnx_load.assert_called()
        mocker_onnx_save.assert_called()
        mocker_postprocess_onnx_model.assert_called()
