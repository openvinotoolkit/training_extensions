# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
import torch
from omegaconf import DictConfig as Config
from otx.v2.adapters.torch.lightning.visual_prompt.engine import VisualPromptEngine
from pytest_mock.plugin import MockerFixture


class TestMMPTEngine:
    def test_init(self, tmp_dir_path: Path) -> None:
        engine = VisualPromptEngine(work_dir=tmp_dir_path)
        assert engine.work_dir == tmp_dir_path

    def test_update_logger(self, tmp_dir_path: Path) -> None:
        engine = VisualPromptEngine(work_dir=tmp_dir_path)
        result = engine._update_logger()
        assert len(result) >= 1
        assert result[0].__class__.__name__ == "CSVLogger"

        result = engine._update_logger(logger=[])
        assert len(result) == 0

        from pytorch_lightning.loggers.logger import DummyLogger
        class MockLogger(DummyLogger):
            def __init__(self) -> None:
                pass

        mock_logger = MockLogger()
        result = engine._update_logger(logger=mock_logger)
        assert result == [mock_logger]

    def test_update_callbacks(self, tmp_dir_path: Path) -> None:
        engine = VisualPromptEngine(work_dir=tmp_dir_path)
        result = engine._update_callbacks()
        assert len(result) == 1

        result = engine._update_callbacks(mode="train_val")
        assert len(result) == 4

        result = engine._update_callbacks(callbacks=[])
        assert result == []

        from pytorch_lightning.callbacks import Callback
        class MockCallback(Callback):
            def __init__(self) -> None:
                pass

        mock_callback = MockCallback()
        result = engine._update_callbacks(callbacks=mock_callback)
        assert result == [mock_callback]

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
            device=None,
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
            device=None,
            logger=False,
            callbacks=None,
        )

    def test_export(self, mocker: MockerFixture, tmp_dir_path: Path) -> None:
        mock_export = mocker.patch("otx.v2.adapters.torch.lightning.visual_prompt.engine.torch.onnx.export")
        mocker.patch("otx.v2.adapters.torch.lightning.visual_prompt.engine.Path.mkdir")
        mocker.patch("otx.v2.adapters.torch.lightning.visual_prompt.engine.Path.exists", return_value=True)
        mock_load_checkpoint = mocker.patch("otx.v2.adapters.torch.lightning.visual_prompt.engine.LightningEngine._load_checkpoint")
        mock_load = mocker.patch("otx.v2.adapters.torch.lightning.visual_prompt.engine.torch.load")
        mock_load.return_value = {"model": {"state_dict": {}}}
        mock_run = mocker.patch("otx.v2.adapters.torch.lightning.visual_prompt.engine.run")
        mock_zeros = mocker.patch("otx.v2.adapters.torch.lightning.visual_prompt.engine.torch.zeros")
        mock_zeros.return_value.to.return_value = mocker.Mock()
        engine = VisualPromptEngine(work_dir=tmp_dir_path)

        mock_model = mocker.Mock()
        mock_model.model.prompt_encoder.embed_dim = 256
        mock_model.model.prompt_encoder.image_embedding_size = (1024 // 16, 1024 //16)
        results = engine.export(model=mock_model, checkpoint="test.pth", precision="16", device="cpu", input_shape=(321, 321))

        export_dir = tmp_dir_path / f"{engine.timestamp}_export"

        mock_load_checkpoint.assert_called()
        mock_export.assert_called()
        mock_run.assert_called()
        assert "outputs" in results
        assert "onnx" in results["outputs"]
        assert "encoder" in results["outputs"]["onnx"]
        assert results["outputs"]["onnx"]["encoder"] == str(export_dir / "onnx" / "sam_encoder.onnx")
        assert "sam" in results["outputs"]["onnx"]
        assert results["outputs"]["onnx"]["sam"] == str(export_dir / "onnx" / "sam.onnx")
        assert "openvino" in results["outputs"]
        assert "encoder" in results["outputs"]["openvino"]
        assert "bin" in results["outputs"]["openvino"]["encoder"]
        assert results["outputs"]["openvino"]["encoder"]["bin"] == str(export_dir / "openvino" / "encoder.bin")
        assert "xml" in results["outputs"]["openvino"]["encoder"]
        assert results["outputs"]["openvino"]["encoder"]["xml"] == str(export_dir / "openvino" / "encoder.xml")
        assert "sam" in results["outputs"]["openvino"]
        assert "bin" in results["outputs"]["openvino"]["sam"]
        assert results["outputs"]["openvino"]["sam"]["bin"] == str(export_dir / "openvino" / "sam.bin")
        assert "xml" in results["outputs"]["openvino"]["sam"]
        assert results["outputs"]["openvino"]["sam"]["xml"] == str(export_dir / "openvino" / "sam.xml")

        mock_load = mocker.patch("otx.v2.adapters.torch.lightning.visual_prompt.engine.torch.load")
        results = engine.export(model=mock_model)
        mock_load.assert_not_called()

        mocker.patch("otx.v2.adapters.torch.lightning.visual_prompt.engine.Path.exists", return_value=False)
        with pytest.raises(RuntimeError, match="OpenVINO Export failed."):
            engine.export(model=mock_model, checkpoint="test.pth", precision="16")
