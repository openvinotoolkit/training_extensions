from typing import List, Optional, Union

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from otx.v2.api.core.engine import Engine
from otx.v2.api.utils.decorators import set_default_argument
from pytorch_lightning import Trainer
from pytorch_lightning.core.datamodule import LightningDataModule
from pytorch_lightning.trainer.connectors.accelerator_connector import (
    _PRECISION_INPUT,
)
from torch.utils.data import DataLoader


class AnomalibEngine(Engine):
    def __init__(
        self,
        work_dir: Optional[str],
    ) -> None:
        super().__init__(work_dir=work_dir)

    def train(
        self,
        model: pl.LightningModule,
        train_dataloader: Union[DataLoader, LightningDataModule],
        val_dataloader: Optional[DataLoader] = None,
        optimizers: Optional[List[torch.optim.Optimizer]] = None,
        max_iters: Optional[int] = None,
        max_epochs: Optional[int] = None,
        distributed: bool = False,
        seed: Optional[int] = None,
        deterministric: Optional[bool] = None,
        precision: _PRECISION_INPUT = 32,
        eval_interval: int = 1,
        eval_metric: Optional[Union[str, List[str]]] = ["accuracy", "class_accuracy"],
        **kwargs,  # Trainer.__init__ arguments
    ):
        configs = DictConfig(kwargs)

        # Set Configuration
        configs.default_root_dir = self.work_dir
        # configs.distributed = distributed
        configs.precision = precision
        if max_epochs is not None:
            configs.max_epochs = max_epochs
            configs.max_steps = -1
        elif max_iters is not None:
            configs.max_epochs = None
            configs.max_steps = max_iters
        if deterministric is not None:
            configs.deterministric = deterministric
        if eval_interval is not None:
            # Validation Interval in Trainer -> val_check_interval
            configs.val_check_interval = eval_interval

        # TODO: Need to re-check
        datamodule = configs.pop("datamodule", None)
        ckpt_path = configs.pop("ckpt_path", None)

        trainer = Trainer(
            **configs,
        )
        if optimizers is not None:
            trainer.optimizers = optimizers

        trainer.fit(
            model=model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
            datamodule=datamodule,
            ckpt_path=ckpt_path,
        )
