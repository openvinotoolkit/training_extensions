# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from otx.v2.adapters.torch.mmengine.mmdeploy.exporter import Exporter
from otx.v2.adapters.torch.mmengine.modules.utils.config_utils import CustomConfig as Config
from pytest_mock.plugin import MockerFixture


class TestExporter:
    @pytest.fixture()
    def exporter(self, mocker: MockerFixture) -> Exporter:
        mocker.patch("otx.v2.adapters.torch.mmengine.mmdeploy.exporter.build_task_processor")
        mocker.patch("otx.v2.adapters.torch.mmengine.mmdeploy.exporter.torch.load", return_value={"model": {"state_dict": {}}})
        mocker.patch("otx.v2.adapters.torch.mmengine.mmdeploy.exporter.torch.nn.Module.load_state_dict")
        config = Config({})
        checkpoint = "checkpoint.pth"
        deploy_config = Config(
            {
                "backend_config": {"mo_options": {}, "model_inputs": [{"opt_shapes": {"input": [224, 224]}}]},
                "ir_config": {"input_names": ["input"], "output_names": ["output"]},
            },
        )
        work_dir = "/path/to/work_dir"
        precision = None
        export_type = "OPENVINO"
        device = "cpu"
        return Exporter(config, checkpoint, deploy_config, work_dir, precision, export_type, device)

    def test_init(self, exporter: Exporter) -> None:
        assert exporter is not None
        assert exporter.work_dir == "/path/to/work_dir"
        assert exporter.onnx_only is False

        config = Config({})
        checkpoint = "checkpoint.pth"
        deploy_config = Config(
            {
                "backend_config": {"mo_options": {}, "model_inputs": [{"opt_shapes": {"input": [224, 224]}}]},
                "ir_config": {"input_names": ["input"], "output_names": ["output"]},
            },
        )
        work_dir = "/path/to/work_dir2"
        precision = "fp16"
        export_type = "onnx"
        device = "cpu"
        exporter = Exporter(config, checkpoint, deploy_config, work_dir, precision, export_type, device)
        assert exporter is not None
        assert exporter.work_dir == work_dir
        assert exporter.deploy_cfg.backend_config.mo_options["flags"] == ["--compress_to_fp16"]
        assert exporter.onnx_only

    def test_export(self, exporter: Exporter, mocker: MockerFixture) -> None:
        mocker.patch("otx.v2.adapters.torch.mmengine.mmdeploy.exporter.export")
        mocker.patch("otx.v2.adapters.torch.mmengine.mmdeploy.exporter.Path.iterdir", return_value=["test.onnx", "openvino.bin", "openvino.xml"])
        mocker.patch("otx.v2.adapters.torch.mmengine.mmdeploy.exporter.Path.mkdir")
        mocker.patch("otx.v2.adapters.torch.mmengine.mmdeploy.exporter.from_onnx")
        results = exporter.export()
        assert results is not None
        assert "outputs" in results
        assert "onnx" in results["outputs"]
        assert "bin" in results["outputs"]
        assert "xml" in results["outputs"]

        exporter.onnx_only = True
        results = exporter.export()
        assert results is not None
        assert "outputs" in results
        assert "onnx" in results["outputs"]
        assert "bin" not in results["outputs"]
        assert "xml" not in results["outputs"]
