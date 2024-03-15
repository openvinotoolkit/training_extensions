"""LossDynamics Mix-in for detection tasks."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from collections import defaultdict
from typing import Dict, Sequence, Tuple

import datumaro as dm
import numpy as np
import pandas as pd

from otx.algorithms.detection.adapters.mmdet.models.loss_dyns import TrackingLossType
from otx.api.entities.dataset_item import DatasetItemEntityWithID
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.shapes.rectangle import Rectangle
from otx.core.data.noisy_label_detection import (
    LossDynamicsTracker,
    LossDynamicsTrackingMixin,
)
from otx.utils.logger import get_logger

logger = get_logger()


class DetLossDynamicsTracker(LossDynamicsTracker):
    """Loss dynamics tracker for detection tasks."""

    TASK_NAME = "OTX-Det"

    def __init__(self, tracking_loss_types: Sequence[TrackingLossType]) -> None:
        super().__init__()
        self._loss_dynamics: Dict[TrackingLossType, Dict] = {
            loss_type: defaultdict(list) for loss_type in tracking_loss_types
        }

    def _convert_anns(self, item: DatasetItemEntityWithID):
        labels = []

        cnt = 0
        for ann in item.get_annotations(preserve_id=True):
            if isinstance(ann.shape, Rectangle):
                for label in ann.get_labels():
                    bbox = dm.Bbox(
                        x=ann.shape.x1 * item.width,
                        y=ann.shape.y1 * item.height,
                        w=ann.shape.width * item.width,
                        h=ann.shape.height * item.height,
                        label=self.otx_label_map[label.id_],
                        id=cnt,
                    )
                    labels.append(bbox)
                    self.otx_ann_id_to_dm_ann_map[(item.id_, ann.id_)] = bbox
                    cnt += 1

        return labels

    def init_with_otx_dataset(self, otx_dataset: DatasetEntity[DatasetItemEntityWithID]) -> None:
        """DatasetEntity should be injected to the tracker for the initialization."""
        self.otx_ann_id_to_dm_ann_map: Dict[Tuple[str, str], dm.Bbox] = {}
        super().init_with_otx_dataset(otx_dataset)

    def accumulate(self, outputs, iter) -> None:
        """Accumulate training loss dynamics for each training step."""
        for key, loss_dyns in outputs.items():
            if isinstance(key, TrackingLossType):
                for (entity_id, ann_id), value in loss_dyns.items():
                    self._loss_dynamics[key][(entity_id, ann_id)].append((iter, value))

    def export(self, output_path: str) -> None:
        """Export loss dynamics statistics to Datumaro format."""
        dfs = [
            pd.DataFrame.from_dict(
                {
                    k: (np.array([iter for iter, _ in arr]), np.array([value for _, value in arr]))
                    for k, arr in loss_dyns.items()
                },
                orient="index",
                columns=["iters", f"loss_dynamics_{key.name}"],
            )
            for key, loss_dyns in self._loss_dynamics.items()
        ]
        df = pd.concat(dfs, axis=1)
        df = df.loc[:, ~df.columns.duplicated()]

        for (entity_id, ann_id), row in df.iterrows():
            ann = self.otx_ann_id_to_dm_ann_map.get((entity_id, ann_id), None)
            if ann:
                ann.attributes = row.to_dict()

        self._export_dataset.export(output_path, format="datumaro")


class DetLossDynamicsTrackingMixin(LossDynamicsTrackingMixin):
    """Mix-in to track loss dynamics during training for classification tasks."""

    TRACKING_LOSS_TYPE: Tuple[TrackingLossType, ...] = ()

    def __init__(self, track_loss_dynamics: bool = False, **kwargs):
        if track_loss_dynamics:
            head_cfg = kwargs.get("bbox_head", None)
            head_type = head_cfg.get("type", None)
            assert head_type is not None, "head_type should be specified from the config."
            new_head_type = head_type + "TrackingLossDynamics"
            head_cfg["type"] = new_head_type
            logger.info(f"Replace head_type from {head_type} to {new_head_type}.")

        super().__init__(**kwargs)

        # This should be called after super().__init__(),
        # since LossDynamicsTrackingMixin.__init__() creates self._loss_dyns_tracker
        self._loss_dyns_tracker = DetLossDynamicsTracker(self.TRACKING_LOSS_TYPE)

    def train_step(self, data, optimizer):
        """The iteration step during training."""

        outputs = super().train_step(data, optimizer)

        if self.loss_dyns_tracker.initialized:
            gt_ann_ids = [item["gt_ann_ids"] for item in data["img_metas"]]

            to_update = {}
            for key, loss_dyns in self.bbox_head.loss_dyns.items():
                to_update[key] = {}
                for (batch_idx, bbox_idx), value in loss_dyns.items():
                    entity_id, ann_id = gt_ann_ids[batch_idx][bbox_idx]
                    to_update[key][(entity_id, ann_id)] = value.mean

            outputs.update(to_update)

        return outputs
