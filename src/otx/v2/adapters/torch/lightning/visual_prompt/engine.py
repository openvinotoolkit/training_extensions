"""OTX adapters.torch.lightning.visual_prompt.Engine API."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Union

import numpy as np
from pytorch_lightning.core.datamodule import LightningDataModule
from torch.utils.data import DataLoader

from otx.v2.adapters.torch.lightning.engine import LightningEngine
from otx.v2.adapters.torch.lightning.model import BaseOTXLightningModel

from .registry import VisualPromptRegistry

if TYPE_CHECKING:
    import pytorch_lightning as pl
    from pytorch_lightning.loggers.logger import Logger
    from pytorch_lightning.utilities.types import EVAL_DATALOADERS

PREDICT_FORMAT = Union[str, Path, np.ndarray]


class VisualPromptEngine(LightningEngine):
    """VisualPrompt engine using PyTorch and PyTorch Lightning."""

    def __init__(
        self,
        work_dir: str | Path | None = None,
        config: str | dict | None = None,
        task: str = "visual_prompting",
    ) -> None:
        """Initialize the VisualPrompt engine.

        Args:
            work_dir (Optional[Union[str, Path]], optional): The working directory for the engine. Defaults to None.
            config (Optional[Union[str, dict]], optional): The configuration for the engine. Defaults to None.
            task (str, optional): The task to perform. Defaults to "classification".
        """
        super().__init__(work_dir=work_dir, config=config, task=task)
        self.registry = VisualPromptRegistry()

    def predict(
        self,
        model: BaseOTXLightningModel | pl.LightningModule | None = None,
        img: PREDICT_FORMAT | (EVAL_DATALOADERS | LightningDataModule) | None = None,
        checkpoint: str | Path | None = None,
        logger: list[Logger] | Logger | bool | None = False,
        callbacks: list[pl.Callback] | pl.Callback | None = None,
        device: str | None = "auto",  # ["auto", "cpu", "gpu", "cuda"]
    ) -> list:
        """Run inference on the given model and input data.

        Args:
            model (Optional[Union[torch.nn.Module, pl.LightningModule]]): The model to use for inference.
            img (Optional[Union[PREDICT_FORMAT, LightningDataModule]]): The input data to run inference on.
            checkpoint (Optional[Union[str, Path]]): The path to the checkpoint file to use for inference.
            device (Optional[list]): The device to use for inference. Can be "auto", "cpu", "gpu", or "cuda".

        Returns:
            list: The output of the inference.
        """
        dataloader = None
        if isinstance(img, (str, Path)):
            from .modules.datasets.dataset import VisualPromptInferenceDataset

            dataset_config = self.config.get("dataset", {})
            image_size = dataset_config.get("image_size", 1024)
            dataset = VisualPromptInferenceDataset(path=img, image_size=image_size)
            dataloader = DataLoader(dataset)
        elif isinstance(img, (DataLoader, LightningDataModule)):
            dataloader = [img]
        return super().predict(
            model=model,
            img=dataloader,
            checkpoint=checkpoint,
            logger=logger,
            callbacks=callbacks,
            device=device,
        )
