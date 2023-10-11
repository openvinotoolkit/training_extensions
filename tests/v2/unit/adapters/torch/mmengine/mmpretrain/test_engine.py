# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
import torch
from mmengine.model import BaseModel
from otx.v2.adapters.torch.mmengine.mmpretrain.engine import MMPTEngine
from otx.v2.adapters.torch.mmengine.modules.utils.config_utils import CustomConfig as Config
from pytest_mock.plugin import MockerFixture


class TestMMPTEngine:
    def test_init(self, tmp_dir_path: Path) -> None:
        engine = MMPTEngine(work_dir=tmp_dir_path)
        assert engine.work_dir == tmp_dir_path

    def test_predict(self, mocker: MockerFixture, tmp_dir_path: Path) -> None:
        mocker.patch("otx.v2.adapters.torch.mmengine.mmpretrain.dataset.get_default_pipeline", return_value=[])
        mock_result = mocker.MagicMock()
        mock_api = mocker.patch("mmpretrain.inference_model", return_value=mock_result)
        mock_inferencer = mocker.patch("mmpretrain.ImageClassificationInferencer")
        engine = MMPTEngine(work_dir=tmp_dir_path)

        class MockModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self._config = Config({})

        engine.predict(model=MockModule(), checkpoint=tmp_dir_path / "weight.pth")
        mock_inferencer.assert_called_once()
        engine.predict(model={"_config": {}})
        mock_inferencer.assert_called()

        class MockModel(BaseModel):
            def __init__(self) -> None:
                super().__init__()
                self._metainfo = Config({"results": [Config({"task": "Image Caption"})]})
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x

        engine.predict(model=MockModel())
        mock_api.assert_called_once()

        with pytest.raises(NotImplementedError):
            engine.predict(model=MockModule(), task="invalid")

    def test_export(self, mocker: MockerFixture, tmp_dir_path: Path) -> None:
        mock_super_export = mocker.patch("otx.v2.adapters.torch.mmengine.mmpretrain.engine.MMXEngine.export")
        mock_super_export.return_value = {"outputs": {"bin": "test.bin", "xml": "test.xml"}}
        engine = MMPTEngine(work_dir=tmp_dir_path)

        result = engine.export(model="model", checkpoint="checkpoint")
        assert result == {"outputs": {"bin": "test.bin", "xml": "test.xml"}}
        mock_super_export.assert_called_once_with(
            model="model", checkpoint="checkpoint", precision="float32", task="Classification", codebase="mmpretrain", export_type="OPENVINO", deploy_config=None, device="cpu", input_shape=None,
        )
