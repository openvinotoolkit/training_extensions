# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Unit tests of visual prompting exporter."""

import pytest
from otx.core.exporter.visual_prompting import OTXVisualPromptingModelExporter
from otx.core.types.export import OTXExportFormatType
from torch import nn


class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = nn.Identity()
        self.embed_dim = 2
        self.image_embedding_size = 4

    def forward(self, x):
        return x


class TestOTXVisualPromptingModelExporter:
    @pytest.fixture()
    def otx_visual_prompting_model_exporter(self) -> OTXVisualPromptingModelExporter:
        return OTXVisualPromptingModelExporter(input_size=(10, 10), via_onnx=True)

    def test_export_openvino(self, mocker, tmpdir, otx_visual_prompting_model_exporter) -> None:
        """Test export for OPENVINO."""
        mocker_torch_onnx_export = mocker.patch("torch.onnx.export")
        mocker_onnx_load = mocker.patch("onnx.load")
        mocker_onnx_save = mocker.patch("onnx.save")
        mocker_postprocess_onnx_model = mocker.patch.object(
            otx_visual_prompting_model_exporter,
            "_postprocess_onnx_model",
        )
        mocker_openvino_convert_model = mocker.patch("openvino.convert_model")
        mocker_postprocess_openvino_model = mocker.patch.object(
            otx_visual_prompting_model_exporter,
            "_postprocess_openvino_model",
        )
        mocker_openvino_save_model = mocker.patch("openvino.save_model")

        otx_visual_prompting_model_exporter.export(
            model=MockModel(),
            output_dir=tmpdir,
            export_format=OTXExportFormatType.OPENVINO,
        )

        mocker_torch_onnx_export.assert_called()
        mocker_onnx_load.assert_called()
        mocker_onnx_save.assert_called()
        mocker_postprocess_onnx_model.assert_called()
        mocker_openvino_convert_model.assert_called()
        mocker_postprocess_openvino_model.assert_called()
        mocker_openvino_save_model.assert_called()

    def test_export_onnx(self, mocker, tmpdir, otx_visual_prompting_model_exporter) -> None:
        """Test export for ONNX."""
        mocker_torch_onnx_export = mocker.patch("torch.onnx.export")
        mocker_onnx_load = mocker.patch("onnx.load")
        mocker_onnx_save = mocker.patch("onnx.save")
        mocker_postprocess_onnx_model = mocker.patch.object(
            otx_visual_prompting_model_exporter,
            "_postprocess_onnx_model",
        )

        otx_visual_prompting_model_exporter.export(
            model=MockModel(),
            output_dir=tmpdir,
            export_format=OTXExportFormatType.ONNX,
        )

        mocker_torch_onnx_export.assert_called()
        mocker_onnx_load.assert_called()
        mocker_onnx_save.assert_called()
        mocker_postprocess_onnx_model.assert_called()

    def test_export_exportable_code(self, tmpdir, otx_visual_prompting_model_exporter) -> None:
        """Test export for EXPORTABLE_CODE."""
        with pytest.raises(NotImplementedError):
            otx_visual_prompting_model_exporter.export(
                model=MockModel(),
                output_dir=tmpdir,
                export_format=OTXExportFormatType.EXPORTABLE_CODE,
            )
