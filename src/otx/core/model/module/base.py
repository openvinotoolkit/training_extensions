# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition for base lightning module used in OTX."""
from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Any

import torch
from lightning import LightningModule
from torch import Tensor

from otx.core.data.entity.base import OTXBatchDataEntity
from otx.core.model.entity.base import OTXModel

if TYPE_CHECKING:
    from otx.core.data.dataset.base import DataMetaInfo


class OTXLitModule(LightningModule):
    """Base class for the lightning module used in OTX."""

    def __init__(
        self,
        otx_model: OTXModel,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        torch_compile: bool,
    ):
        super().__init__()

        self.model = otx_model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.torch_compile = torch_compile

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["otx_model"])

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
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": self.lr_scheduler_monitor_key,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    def register_load_state_dict_pre_hook(self, model_classes: list[str], ckpt_classes: list[str]) -> None:
        """Register self.model's load_state_dict_pre_hook.

        Args:
            model_classes (list[str]): Class names from training data.
            ckpt_classes (list[str]): Class names from checkpoint state dictionary.
        """
        self.model.register_load_state_dict_pre_hook(model_classes, ckpt_classes)

    def state_dict(self) -> dict[str, Any]:
        """Return state dictionary of model entity with meta information.

        Returns:
            A dictionary containing datamodule state.

        """
        state_dict = super().state_dict()
        
        def detach_complex_prefix(state_dict: dict[str, Any]) -> None:
            """Detach the model.model prefix to make more readable."""
            for key, value in state_dict.items():
                if key.startswith("model.model."):
                    new_key = key.replace("model.model.", "", 1)
                    state_dict[new_key] = value
        detach_complex_prefix(state_dict)
        state_dict["meta_info"] = self.meta_info
        return state_dict

    def load_state_dict(self, state_dict: dict[str, Any], *args, **kwargs) -> None:
        """Load state dictionary from checkpoint state dictionary.

        If checkpoint's meta_info and OTXLitModule's meta_info are different,
        load_state_pre_hook for smart weight loading will be registered.
        """
        ckpt_meta_info = state_dict.pop("meta_info", None)

        if ckpt_meta_info and self.meta_info is None:
            msg = (
                "`state_dict` to load has `meta_info`, but the current model has no `meta_info`. "
                "It is recommended to set proper `meta_info` for the incremental learning case."
            )
            warnings.warn(msg, stacklevel=2)
        if ckpt_meta_info and self.meta_info and ckpt_meta_info != self.meta_info:
            logger = logging.getLogger()
            logger.info(
                f"Data classes from checkpoint: {ckpt_meta_info.class_names} -> "
                f"Data classes from training data: {self.meta_info.class_names}",
            )
            self.register_load_state_dict_pre_hook(
                self.meta_info.class_names,
                ckpt_meta_info.class_names,
            )
        return super().load_state_dict(state_dict, *args, **kwargs)

    @property
    def lr_scheduler_monitor_key(self) -> str:
        """Metric name that the learning rate scheduler monitor."""
        return "val/loss"

    @property
    def meta_info(self) -> DataMetaInfo:
        """Meta information of OTXLitModule."""
        if self._meta_info is None:
            err_msg = "meta_info is referenced before assignment"
            raise ValueError(err_msg)
        return self._meta_info

    @meta_info.setter
    def meta_info(self, meta_info: DataMetaInfo) -> None:
        self._meta_info = meta_info
