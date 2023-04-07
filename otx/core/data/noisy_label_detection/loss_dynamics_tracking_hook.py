"""Hook module to track loss dynamics during training and export these statistics to Datumaro format."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import os.path as osp

from mmcv.parallel import MMDataParallel
from mmcv.runner import BaseRunner
from mmcv.runner.hooks import HOOKS, Hook

from otx.algorithms.common.utils.logger import get_logger
from otx.api.entities.datasets import DatasetEntity

from .base import LossDynamicsTracker, LossDynamicsTrackingMixin

logger = get_logger()

__all__ = ["LossDynamicsTrackingHook"]


@HOOKS.register_module()
class LossDynamicsTrackingHook(Hook):
    """Tracking loss dynamics during training and export it to Datumaro dataset format."""

    def __init__(self, output_path: str, alpha: float = 0.001) -> None:
        self._output_fpath = osp.join(output_path, "noisy_label_detection")

    def before_run(self, runner):
        """Before run, check the type of model for safe running."""
        if not isinstance(runner.model, MMDataParallel):
            raise NotImplementedError(f"Except MMDataParallel, runner.model={type(runner.model)} is not supported now.")

    def before_train_epoch(self, runner: BaseRunner):
        """Initialize the tracker for training loss dynamics tracking.

        Tracker needs the training dataset for initialization.
        However, there is no way to access to dataloader until the beginning of training epoch.
        """
        self._init_tracker(runner, runner.data_loader.dataset.otx_dataset)

    def _get_tracker(self, runner: BaseRunner) -> LossDynamicsTracker:
        model = runner.model.module

        if not isinstance(model, LossDynamicsTrackingMixin):
            raise RuntimeError(
                f"The model should be an instance of LossDynamicsTrackingMixin, but type(model)={type(model)}."
            )
        return model.loss_dyns_tracker

    def _init_tracker(self, runner: BaseRunner, otx_dataset: DatasetEntity) -> None:
        tracker = self._get_tracker(runner)
        if tracker.initialized:
            return

        logger.info("Initialize training loss dynamics tracker.")
        tracker.init_with_otx_dataset(otx_dataset)

    def after_train_iter(self, runner):
        """Accumulate training loss dynamics.

        It should be here because it needs to access the training iteration.
        """
        tracker = self._get_tracker(runner)
        tracker.accumulate(runner.outputs, runner.iter)

    def after_run(self, runner: BaseRunner) -> None:
        """Export loss dynamics statistics to Datumaro format."""

        tracker = self._get_tracker(runner)

        if tracker.initialized:
            logger.info(f"Export training loss dynamics to {self._output_fpath}")
            tracker.export(self._output_fpath)

    @classmethod
    def configure_train_pipeline(cls, data_cfg) -> None:
        """Configure data pipeline to include "entity_id".

        This should be called before building the train dataloader.
        """
        for p in data_cfg.train.pipeline:
            if p.type == "Collect":
                p.meta_keys.add("entity_id")
