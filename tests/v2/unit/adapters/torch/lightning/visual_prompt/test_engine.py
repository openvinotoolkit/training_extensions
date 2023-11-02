# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
import torch
from omegaconf import DictConfig as Config
from otx.v2.adapters.torch.lightning.visual_prompt.engine import VisualPromptEngine
from pytest_mock.plugin import MockerFixture


class TestVisualPromptEngine:
    def test_init(self, tmp_dir_path: Path) -> None:
        engine = VisualPromptEngine(work_dir=tmp_dir_path)
        assert engine.work_dir == tmp_dir_path

    def test_update_logger(self, tmp_dir_path: Path) -> None:
        engine = VisualPromptEngine(work_dir=tmp_dir_path)
        result = engine._update_logger()
        assert len(result) >= 1
        assert result[0].__class__.__name__ == "CSVLogger"

        result = engine._update_logger(logger=[])
        assert len(result) == 1
        assert result[0].__class__.__name__ == "CSVLogger"

        from pytorch_lightning.loggers.logger import DummyLogger
        class MockLogger(DummyLogger):
            def __init__(self) -> None:
                pass

        mock_logger = MockLogger()
        result = engine._update_logger(logger=[mock_logger])
        assert len(result) == 2

    def test_predict(self, mocker: MockerFixture, tmp_dir_path: Path) -> None:
        mock_super_predict = mocker.patch("otx.v2.adapters.torch.lightning.visual_prompt.engine.LightningEngine.predict")
        engine = VisualPromptEngine(work_dir=tmp_dir_path)

        class MockModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self._config = Config({})
        mock_model = MockModule()

        # img -> DataLoader
        from torch.utils.data import DataLoader
        class MockDataloader(DataLoader):
            def __init__(self) -> None:
                pass

        mock_dataloader = MockDataloader()
        engine.predict(
            model=mock_model,
            img=mock_dataloader,
        )
        mock_super_predict.assert_called_once_with(
            model=mock_model,
            img=[mock_dataloader],
            checkpoint = None,
            device="auto",
            logger=False,
            callbacks=None,
        )

        # img -> str
        mock_inference_dataset = mocker.patch("otx.v2.adapters.torch.lightning.visual_prompt.modules.datasets.dataset.VisualPromptInferenceDataset")
        mock_torch_dataloader = mocker.patch("otx.v2.adapters.torch.lightning.visual_prompt.engine.DataLoader")
        engine.predict(
            model=mock_model,
            img="test.png",
            checkpoint=tmp_dir_path / "weight.pth",
        )
        mock_inference_dataset.assert_called_once_with(
            path="test.png",
            image_size=1024,
        )
        mock_torch_dataloader.assert_called_once_with(
            mock_inference_dataset.return_value,
        )
        mock_super_predict.assert_called_with(
            model=mock_model,
            img=mock_torch_dataloader.return_value,
            checkpoint =tmp_dir_path / "weight.pth",
            device="auto",
            logger=False,
            callbacks=None,
        )
