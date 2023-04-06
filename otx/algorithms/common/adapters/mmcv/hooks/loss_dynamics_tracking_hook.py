"""Hook module to track loss dynamics during training and export these statistics to Datumaro format."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import os.path as osp
from collections import defaultdict
from typing import Any, Dict, List, Union, get_args

import datumaro as dm
import numpy as np
import pandas as pd
from mmcv.runner import BaseRunner
from mmcv.runner.hooks import HOOKS, Hook

from otx.algorithms.classification.adapters.mmcls import OTXClsDataset
from otx.algorithms.common.utils.logger import get_logger
from otx.api.entities.dataset_item import DatasetItemEntityWithID
from otx.api.entities.datasets import DatasetEntity

logger = get_logger()

# TODO: More than two types are needed to create Union so that None is inserted here.
# This space is reserved for OTXDetDataset in the future.
_ALLOWED_DATASET_TYPES = Union[OTXClsDataset, None]


@HOOKS.register_module()
class LossDynamicsTrackingHook(Hook):
    """Tracking loss dynamics during training and export it to Datumaro dataset format."""

    def __init__(self, output_path: str, alpha: float = 0.001) -> None:
        self._loss_dynamics: Dict[Any, List] = defaultdict(list)
        self._output_fpath = osp.join(output_path, "loss_dynamics")
        self._alpha = alpha

    def before_train_epoch(self, runner: BaseRunner):
        """Initialization for training loss dynamics tracking."""
        if not isinstance(runner.data_loader.dataset, get_args(_ALLOWED_DATASET_TYPES)):
            raise RuntimeError(f"{type(runner.data_loader.dataset)} is not allowed.")

        dataset = runner.data_loader.dataset

        if isinstance(dataset, OTXClsDataset):
            self._init_cls_task(dataset.otx_dataset)

    def after_train_iter(self, runner: BaseRunner) -> None:
        """Accumulate training loss dynamics for each training step."""
        entity_ids = runner.outputs["entity_ids"]
        gt_labels = np.squeeze(runner.outputs["gt_labels"])
        loss_dyns = runner.outputs["loss_dyns"]

        for entity_id, gt_label, loss_dyn in zip(entity_ids, gt_labels, loss_dyns):
            self._loss_dynamics[(entity_id, gt_label)].append((runner.iter, loss_dyn))

    def after_run(self, runner: BaseRunner) -> None:
        """Export loss dynamics statistics to Datumaro format."""
        df = pd.DataFrame.from_dict(
            {
                k: (np.array([iter for iter, _ in arr]), np.array([value for _, value in arr]))
                for k, arr in self._loss_dynamics.items()
            },
            orient="index",
            columns=["iters", "loss_dynamics"],
        )

        for (entity_id, gt_label), row in df.iterrows():
            item = self._export_dataset.get(entity_id, "train")
            for ann in item.annotations:
                if isinstance(ann, dm.Label) and ann.label == gt_label:
                    ann.attributes = row.to_dict()

        self._export_dataset.export(self._output_fpath, format="datumaro")
        logger.info(f"Export training loss dynamics to {self._output_fpath}")

    def _init_cls_task(self, otx_dataset: DatasetEntity):
        otx_labels = otx_dataset.get_labels()
        label_categories = dm.LabelCategories.from_iterable([label_entity.name for label_entity in otx_labels])
        otx_label_map = {label_entity.id_: idx for idx, label_entity in enumerate(otx_labels)}

        def _convert_anns(item: DatasetItemEntityWithID):
            labels = [
                dm.Label(label=otx_label_map[label.id_]) for ann in item.get_annotations() for label in ann.get_labels()
            ]
            return labels

        self._export_dataset = dm.Dataset.from_iterable(
            [
                dm.DatasetItem(
                    id=item.id_,
                    subset="train",
                    media=dm.Image(path=item.media.path, size=(item.media.height, item.media.width)),
                    annotations=_convert_anns(item),
                )
                for item in otx_dataset
            ],
            infos={
                "purpose": "To export training loss dynamics for classification tasks",
                "ema_alpha": self._alpha,
            },
            categories={dm.AnnotationType.label: label_categories},
        )
