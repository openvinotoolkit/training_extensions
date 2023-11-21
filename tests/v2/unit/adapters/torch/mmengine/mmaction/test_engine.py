"""Unit-test for the engine API for MMAction."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any

import pytest
import torch
from mmengine.model import BaseModel
from otx.v2.adapters.torch.mmengine.mmaction.engine import MMActionEngine
from otx.v2.adapters.torch.mmengine.modules.utils.config_utils import CustomConfig as Config
from pytest_mock.plugin import MockerFixture


class TestMMActionEngine:
    def test_init(self, tmp_dir_path: Path) -> None:
        engine = MMActionEngine(work_dir=tmp_dir_path)
        assert engine.work_dir == tmp_dir_path

    def test_predict(self, mocker: MockerFixture, tmp_dir_path: Path) -> None:
        mocker.patch("otx.v2.adapters.torch.mmengine.mmaction.dataset.get_default_pipeline", return_value=[])
        mock_inferencer = mocker.patch("mmaction.apis.inferencers.MMAction2Inferencer")
        engine = MMActionEngine(work_dir=tmp_dir_path)

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

        with pytest.raises(NotImplementedError):
            engine.predict(model=MockModule(), task="invalid")

    def test_export(self, mocker: MockerFixture, tmp_dir_path: Path) -> None:
        engine = MMActionEngine(work_dir=tmp_dir_path)
        output_dict = {"outputs": {"bin": "test.bin", "xml": "test.xml"}}
        class MockExporter:
            def export(self, **kwargs):
                self.kwargs = kwargs
                return output_dict
        class MockModule:
            def __call__(self, **kwargs):
                self.kwargs = kwargs
                self.exporter = MockExporter()
                return self.exporter
        mocker.patch("otx.v2.adapters.torch.mmengine.mmdeploy.exporter.Exporter", MockModule())
        result = engine.export(model=Config({"type": ""}), checkpoint="checkpoint")
        assert result == output_dict
