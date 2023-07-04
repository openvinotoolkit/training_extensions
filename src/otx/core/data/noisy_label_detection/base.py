"""Hook module to track loss dynamics during training and export these statistics to Datumaro format."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from typing import List, Optional

import datumaro as dm

from otx.api.entities.dataset_item import DatasetItemEntityWithID
from otx.api.entities.datasets import DatasetEntity

__all__ = ["LossDynamicsTracker", "LossDynamicsTrackingMixin"]


class LossDynamicsTracker:
    """Class to track loss dynamics and export it to Datumaro format."""

    TASK_NAME: Optional[str] = None

    def __init__(self) -> None:
        self.initialized = False

    def init_with_otx_dataset(self, otx_dataset: DatasetEntity) -> None:
        """DatasetEntity should be injected to the tracker for the initialization."""
        otx_labels = otx_dataset.get_labels()
        label_categories = dm.LabelCategories.from_iterable([label_entity.name for label_entity in otx_labels])
        self.otx_label_map = {label_entity.id_: idx for idx, label_entity in enumerate(otx_labels)}

        self._export_dataset = dm.Dataset.from_iterable(
            [
                dm.DatasetItem(
                    id=item.id_,
                    subset="train",
                    media=dm.Image.from_file(path=item.media.path, size=(item.media.height, item.media.width))
                    if item.media.path
                    else dm.Image.from_numpy(
                        data=getattr(item.media, "_Image__data"), size=(item.media.height, item.media.width)
                    ),
                    annotations=self._convert_anns(item),
                )
                for item in otx_dataset
            ],
            infos={"purpose": "noisy_label_detection", "task": self.TASK_NAME},
            categories={dm.AnnotationType.label: label_categories},
        )

        self.initialized = True

    def _convert_anns(self, item: DatasetItemEntityWithID) -> List[dm.Annotation]:
        raise NotImplementedError()

    def accumulate(self, outputs, iter) -> None:
        """Accumulate training loss dynamics for each training step."""
        raise NotImplementedError()

    def export(self, output_path: str) -> None:
        """Export loss dynamics statistics to Datumaro format."""
        raise NotImplementedError()


class LossDynamicsTrackingMixin:
    """Mix-in to track loss dynamics during training."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._loss_dyns_tracker = LossDynamicsTracker()

    @property
    def loss_dyns_tracker(self) -> LossDynamicsTracker:
        """Get tracker."""
        return self._loss_dyns_tracker
