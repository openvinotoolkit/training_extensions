# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Unit test for mmdeploy exporter."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from otx.core.exporter import mmdeploy as target_file
from otx.core.exporter.mmdeploy import (
    MMdeployExporter,
    load_mmconfig_from_pkg,
    mmdeploy_init_model_helper,
    patch_input_shape,
    use_temporary_default_scope,
)
from otx.core.types.precision import OTXPrecisionType


class TestMMdeployExporter:
    DEFAULT_MMDEPLOY_CFG = "otx.algo.detection.mmdeploy.atss"

    @pytest.fixture(autouse=True)
    def setup(self, mocker):
        mocker.patch.object(target_file, "convert_conf_to_mmconfig_dict", return_value=MagicMock())

    def get_exporter(
        self,
        max_num_detections: int | None = 0,
        output_names: list[str] | None = None,
    ) -> MMdeployExporter:
        return MMdeployExporter(
            model_builder=MagicMock(),
            model_cfg=MagicMock(),
            deploy_cfg=self.DEFAULT_MMDEPLOY_CFG,
            test_pipeline=MagicMock(),
            input_size=(1, 3, 256, 256),
            max_num_detections=max_num_detections,
            output_names=output_names,
        )

    def test_init(self):
        max_num_detections = 10
        output_names = ["box"]
        exporter = self.get_exporter(max_num_detections, output_names)

        # check attributes are set well
        for val in output_names:
            assert val in exporter.output_names
        assert (
            exporter._deploy_cfg["codebase_config"]["post_processing"]["max_output_boxes_per_class"]
            == max_num_detections
        )
        assert exporter._deploy_cfg["codebase_config"]["post_processing"]["keep_top_k"] == max_num_detections
        assert exporter._deploy_cfg["codebase_config"]["post_processing"]["pre_top_k"] == max_num_detections * 10

    @pytest.fixture()
    def exported_model(self) -> MagicMock:
        return MagicMock()

    @pytest.fixture()
    def mock_openvino(self, mocker, exported_model) -> MagicMock:
        mock_openvino = mocker.patch.object(target_file, "openvino")
        mock_openvino.convert_model.return_value = exported_model
        return mock_openvino

    @pytest.fixture()
    def onnx_path(self) -> MagicMock:
        return MagicMock()

    @pytest.fixture()
    def mock_cvt2onnx(self, mocker, onnx_path) -> MagicMock:
        return mocker.patch.object(MMdeployExporter, "_cvt2onnx", return_value=onnx_path)

    @pytest.fixture()
    def mock_postprocess_openvino_model(self, mocker) -> MagicMock:
        return mocker.patch.object(
            MMdeployExporter,
            "_postprocess_openvino_model",
            side_effect=lambda x: x,
        )

    @pytest.fixture()
    def mock_model(self) -> MagicMock:
        return MagicMock()

    @pytest.fixture()
    def output_dir(self) -> MagicMock:
        return MagicMock()

    @pytest.fixture()
    def base_model_name(self) -> str:
        return "fake"

    @pytest.fixture()
    def save_path(self, output_dir, base_model_name) -> MagicMock:
        return output_dir / (base_model_name + ".xml")

    @pytest.mark.parametrize("precision", [OTXPrecisionType.FP16, OTXPrecisionType.FP32])
    def test_to_openvino(
        self,
        precision,
        exported_model,
        mock_openvino,
        onnx_path,
        mock_cvt2onnx,
        mock_postprocess_openvino_model,
        mock_model,
        output_dir,
        base_model_name,
        save_path,
    ):
        exporter = self.get_exporter()

        assert save_path == exporter.to_openvino(mock_model, output_dir, base_model_name, precision)

        mock_cvt2onnx.assert_called_once_with(mock_model, output_dir, base_model_name)
        mock_openvino.convert_model.assert_called_once()
        assert mock_openvino.convert_model.call_args.args[0] == str(onnx_path)
        mock_postprocess_openvino_model.assert_called_once_with(exported_model)
        mock_openvino.save_model.assert_called_once_with(
            exported_model,
            save_path,
            compress_to_fp16=(precision == OTXPrecisionType.FP16),
        )
        onnx_path.unlink.assert_called_once()

    @pytest.fixture()
    def onnx_model(self) -> MagicMock:
        return MagicMock()

    @pytest.fixture()
    def mock_onnx(self, mocker, onnx_model) -> MagicMock:
        mock_onnx = mocker.patch.object(target_file, "onnx")
        mock_onnx.load.return_value = onnx_model
        return mock_onnx

    @pytest.fixture()
    def mock_postprocess_onnx_model(self, mocker) -> MagicMock:
        def func(*args):  # noqa: ANN202
            return args[0]

        return mocker.patch.object(MMdeployExporter, "_postprocess_onnx_model", side_effect=func)

    @pytest.mark.parametrize("precision", [OTXPrecisionType.FP16, OTXPrecisionType.FP32])
    def test_to_onnx(
        self,
        mock_model,
        output_dir,
        base_model_name,
        precision,
        onnx_path,
        mock_cvt2onnx,
        mock_onnx,
        onnx_model,
        mock_postprocess_onnx_model,
    ):
        exporter = self.get_exporter()

        assert onnx_path == exporter.to_onnx(mock_model, output_dir, base_model_name, precision)

        mock_cvt2onnx.assert_called_once()
        assert mock_cvt2onnx.call_args.args[0] == mock_model
        assert mock_cvt2onnx.call_args.args[1] == output_dir
        assert mock_cvt2onnx.call_args.args[2] == base_model_name
        assert mock_cvt2onnx.call_args.args[3]["backend_config"]["type"] == "onnxruntime"
        mock_onnx.load.assert_called_once_with(str(onnx_path))
        mock_postprocess_onnx_model.assert_called_once()
        assert mock_postprocess_onnx_model.call_args.args[0] == onnx_model
        mock_onnx.save.assert_called_once_with(onnx_model, str(onnx_path))

    @pytest.fixture()
    def mock_torch(self, mocker) -> MagicMock:
        return mocker.patch.object(target_file, "torch")

    def test_cvt2onnx(self, mocker, mock_model, output_dir, base_model_name, mock_torch):
        exporter = self.get_exporter()
        mock_torch2onnx = mocker.patch.object(target_file, "torch2onnx")
        mock_build_task_processor = mocker.patch.object(target_file, "build_task_processor")
        mock_use_temporary_default_scope = mocker.patch.object(target_file, "use_temporary_default_scope")

        assert output_dir / f"{base_model_name}.onnx" == exporter._cvt2onnx(mock_model, output_dir, base_model_name)
        mock_torch.save.assert_called_once()
        assert mock_torch.save.call_args.args[0] == mock_model.state_dict()
        mock_use_temporary_default_scope.assert_called_once()
        mock_build_task_processor.assert_called_once()
        mock_torch2onnx.assert_called_once()
        assert mock_torch2onnx.call_args.args[1] == str(output_dir)
        assert mock_torch2onnx.call_args.kwargs["model_checkpoint"] == str(mock_torch.save.call_args.args[1])


def test_mmdeploy_init_model_helper():
    mock_model = MagicMock()
    model_parameters = [MagicMock() for _ in range(3)]
    mock_model.parameters.return_value = model_parameters
    mock_model_builder = MagicMock(return_value=mock_model)

    assert mock_model == mmdeploy_init_model_helper(model_builder=mock_model_builder)
    for param in model_parameters:
        assert param.requires_grad is False


def test_patch_input_shape():
    mock_deploy_cfg = MagicMock()
    width = 128
    height = 256

    patch_input_shape(mock_deploy_cfg, width, height)

    assert mock_deploy_cfg.ir_config.input_shape == (width, height)


def test_load_mmconfig_from_pkg(mocker):
    mock_mmconfig = mocker.patch.object(target_file, "MMConfig")
    assert mock_mmconfig.fromfile.return_value == load_mmconfig_from_pkg("otx")
    mock_mmconfig.fromfile.assert_called_once()
    assert "otx/__init__.py" in mock_mmconfig.fromfile.call_args.args[0]


def test_use_temporary_default_scope(mocker):
    mock_default_scope = mocker.patch.object(target_file, "DefaultScope")
    mock_default_scope._instance_dict = {}

    with use_temporary_default_scope():
        mock_default_scope._instance_dict["abc"] = 1

    assert mock_default_scope._instance_dict == {}
