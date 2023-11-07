# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from omegaconf import DictConfig
from otx.v2.adapters.torch.lightning.modules.models.base_model import BaseOTXLightningModel
from pytest_mock.plugin import MockerFixture


class TestBaseOTXLightningModel:
    def test_export(self, mocker: MockerFixture, tmp_dir_path: Path) -> None:
            mock_export = mocker.patch("otx.v2.adapters.torch.lightning.model.torch.onnx.export")
            mocker.patch("otx.v2.adapters.torch.lightning.model.Path.mkdir")
            mocker.patch("otx.v2.adapters.torch.lightning.model.Path.exists", return_value=True)
            mock_run = mocker.patch("subprocess.run")
            mock_zeros = mocker.patch("otx.v2.adapters.torch.lightning.model.torch.zeros")
            mock_zeros.return_value.to.return_value = mocker.Mock()
            class MockModel(BaseOTXLightningModel):
                def callbacks(self) -> list:
                    return []

            model = MockModel()
            model.config = DictConfig({"model": {}})
            model.device = "cpu"

            export_dir = tmp_dir_path / "export_test"
            results = model.export(export_dir=export_dir, precision="16")

            mock_export.assert_called_once_with(
                model=model,
                args=mock_zeros.return_value.to.return_value,
                f=str(export_dir / "onnx" / "onnx_model.onnx"),
                opset_version=11,
            )
            mock_run.assert_called_once_with(
                args=[
                    "mo",
                    "--input_model",
                    str(export_dir / "onnx" / "onnx_model.onnx"),
                    "--output_dir",
                    str(export_dir / "openvino"),
                    "--model_name", "openvino",
                    "--compress_to_fp16",
                ],
                check=False,
            )
            assert "outputs" in results
            assert "onnx" in results["outputs"]
            assert results["outputs"]["onnx"] == str(export_dir / "onnx" / "onnx_model.onnx")
            assert "bin" in results["outputs"]
            assert results["outputs"]["bin"] == str(export_dir / "openvino" / "openvino.bin")
            assert "xml" in results["outputs"]
            assert results["outputs"]["xml"] == str(export_dir / "openvino" / "openvino.xml")

            mocker.patch("otx.v2.adapters.torch.lightning.engine.Path.exists", return_value=False)
            with pytest.raises(RuntimeError, match="OpenVINO Export failed."):
                model.export(export_dir=export_dir, precision="16")
