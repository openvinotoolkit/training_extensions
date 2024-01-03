# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition for base lightning module used in OTX."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import yaml
from lightning import LightningModule
from omegaconf import DictConfig, OmegaConf
from torch import Tensor

from otx.core.data.entity.base import OTXBatchDataEntity
from otx.core.model.entity.base import OTXModel
from otx.core.utils.instantiators import instantiate_model

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable


class OTXLitModule(LightningModule):
    """Base class for the lightning module used in OTX."""

    def __init__(
        self,
        otx_model: OTXModel,
        torch_compile: bool,
        optimizer: OptimizerCallable = lambda p: torch.optim.SGD(p, lr=0.01),
        scheduler: LRSchedulerCallable = torch.optim.lr_scheduler.ConstantLR,
    ):
        super().__init__()

        self.model = otx_model
        self.torch_compile = torch_compile
        self.optimizer = optimizer
        self.scheduler = scheduler

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["otx_model"])

    @classmethod
    def from_config(
        cls,
        config: dict | DictConfig | str | Path,
        overrides: dict | None = None,
    ) -> OTXLitModule:
        """Create an instance of OTXLitModule from a configuration file.

        Args:
            config (dict | str | Path): Path to the configuration file.

        Returns:
            OTXLitModule: An instance of OTXLitModule.

        """
        if isinstance(config, (str, Path)):
            with Path(config).open() as f:
                config = yaml.safe_load(f)
        if not isinstance(config, (dict, DictConfig)):
            msg = "Please double-check model config."
            raise TypeError(msg)
        model_cfg = config.get("model", config)

        # Override model configuration
        if overrides is not None:
            model_cfg = DictConfig(model_cfg)
            model_cfg.merge_with(overrides)

        # Instantiate model
        if isinstance(model_cfg, DictConfig):
            model_cfg = OmegaConf.to_container(model_cfg)
        model = instantiate_model(model_cfg=model_cfg)
        if model is None:
            msg = "Please double-check model config."
            raise TypeError(msg)
        return model

    def training_step(self, inputs: OTXBatchDataEntity, batch_idx: int) -> Tensor:
        """Step for model training."""
        train_loss = self.model(inputs)

        if isinstance(train_loss, Tensor):
            self.log(
                "train/loss",
                train_loss,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
            )
            return train_loss
        if isinstance(train_loss, dict):
            for k, v in train_loss.items():
                self.log(
                    f"train/{k}",
                    v,
                    on_step=True,
                    on_epoch=False,
                    prog_bar=True,
                )

            total_train_loss = sum(train_loss.values())
            self.log(
                "train/loss",
                total_train_loss,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
            )
            return total_train_loss

        raise TypeError(train_loss)

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate, test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.torch_compile and stage == "fit":
            self.model = torch.compile(self.model)

    def configure_optimizers(self) -> dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.

        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.optimizer(self.parameters())
        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer)
            monitor = getattr(scheduler, "monitor", None)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": monitor,
                },
            }
        return {"optimizer": optimizer}
