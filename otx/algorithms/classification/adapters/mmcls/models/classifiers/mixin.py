"""Module defining Mix-in class of SAMClassifier."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from collections import defaultdict
from typing import Any, Dict, List

import datumaro as dm
import numpy as np
import pandas as pd

from otx.algorithms.common.utils.logger import get_logger
from otx.api.entities.dataset_item import DatasetItemEntityWithID
from otx.core.data.noisy_label_detection import (
    LossDynamicsTracker,
    LossDynamicsTrackingMixin,
)

logger = get_logger()


class SAMClassifierMixin:
    """SAM-enabled BaseClassifier mix-in."""

    def train_step(self, data, optimizer=None, **kwargs):
        """Saving current batch data to compute SAM gradient."""
        self.current_batch = data
        return super().train_step(data, optimizer, **kwargs)


class MultiClassClsLossDynamicsTracker(LossDynamicsTracker):
    """Loss dynamics tracker for multi-class classification task."""

    TASK_NAME = "OTX-MultiClassCls"

    def __init__(self) -> None:
        super().__init__()
        self._loss_dynamics: Dict[Any, List] = defaultdict(list)

    def _convert_anns(self, item: DatasetItemEntityWithID):
        labels = [
            dm.Label(label=self.otx_label_map[label.id_])
            for ann in item.get_annotations()
            for label in ann.get_labels()
        ]
        return labels

    def accumulate(self, outputs, iter) -> None:
        """Accumulate training loss dynamics for each training step."""
        entity_ids = outputs["entity_ids"]
        label_ids = np.squeeze(outputs["label_ids"])
        loss_dyns = outputs["loss_dyns"]

        for entity_id, label_id, loss_dyn in zip(entity_ids, label_ids, loss_dyns):
            self._loss_dynamics[(entity_id, label_id)].append((iter, loss_dyn))

    def export(self, output_path: str) -> None:
        """Export loss dynamics statistics to Datumaro format."""
        df = pd.DataFrame.from_dict(
            {
                k: (np.array([iter for iter, _ in arr]), np.array([value for _, value in arr]))
                for k, arr in self._loss_dynamics.items()
            },
            orient="index",
            columns=["iters", "loss_dynamics"],
        )

        for (entity_id, label_id), row in df.iterrows():
            item = self._export_dataset.get(entity_id, "train")
            for ann in item.annotations:
                if isinstance(ann, dm.Label) and ann.label == self.otx_label_map[label_id]:
                    ann.attributes = row.to_dict()

        self._export_dataset.export(output_path, format="datumaro")


class ClsLossDynamicsTrackingMixin(LossDynamicsTrackingMixin):
    """Mix-in to track loss dynamics during training for classification tasks."""

    def __init__(self, track_loss_dynamics: bool = False, **kwargs):
        if track_loss_dynamics:
            if getattr(self, "multilabel", False) or getattr(self, "hierarchical", False):
                raise RuntimeError("multilabel or hierarchical tasks are not supported now.")

            head_cfg = kwargs.get("head", None)
            loss_cfg = head_cfg.get("loss", None)
            loss_cfg["reduction"] = "none"

        # This should be called after modifying "reduction" config.
        super().__init__(**kwargs)

        # This should be called after super().__init__(),
        # since LossDynamicsTrackingMixin.__init__() creates self._loss_dyns_tracker
        self._loss_dyns_tracker = MultiClassClsLossDynamicsTracker()

    def train_step(self, data, optimizer=None, **kwargs):
        """The iteration step for training.

        If self._track_loss_dynamics = False, just follow BaseClassifier.train_step().
        Otherwise, it steps with tracking loss dynamics.
        """
        if self.loss_dyns_tracker.initialized:
            return self._train_step_with_tracking(data, optimizer, **kwargs)
        return super().train_step(data, optimizer, **kwargs)

    def _train_step_with_tracking(self, data, optimizer=None, **kwargs):
        losses = self(**data)

        loss_dyns = losses["loss"].detach().cpu().numpy()
        assert not np.isscalar(loss_dyns)

        entity_ids = [img_meta["entity_id"] for img_meta in data["img_metas"]]
        label_ids = [img_meta["label_id"] for img_meta in data["img_metas"]]
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            loss_dyns=loss_dyns,
            entity_ids=entity_ids,
            label_ids=label_ids,
            num_samples=len(data["img"].data),
        )

        return outputs
