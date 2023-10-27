"""Inference Callbacks for OTX inference."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from bson import ObjectId
from pytorch_lightning.callbacks import Callback

from otx.v2.api.entities.annotation import Annotation
from otx.v2.api.entities.datasets import DatasetEntity
from otx.v2.api.entities.id import ID
from otx.v2.api.entities.image import Image
from otx.v2.api.entities.scored_label import ScoredLabel
from otx.v2.api.entities.utils.segmentation_utils import (
    create_annotation_from_segmentation_map,
    create_hard_prediction_from_soft_prediction,
)

if TYPE_CHECKING:
    from pytorch_lightning import LightningModule, Trainer


class InferenceCallback(Callback):
    """Callback that updates otx_dataset during inference."""

    def __init__(self, otx_dataset: DatasetEntity) -> None:
        """Initializes the InferenceCallback object.

        Args:
            otx_dataset (DatasetEntity): The dataset to use for inference.

        Returns:
            None
        """
        if any(isinstance(shape, Image) for shape in otx_dataset[0].annotation_scene.shapes):
            self.use_mask = True
        else:
            self.use_mask = False
        self.otx_dataset = otx_dataset.with_empty_annotations()

    def on_predict_epoch_end(self, _trainer: Trainer, _pl_module: LightningModule, outputs: list) -> None:
        """Called at the end of the prediction epoch.

        Args:
            _trainer: The trainer object.
            _pl_module: The lightning module being trained.
            outputs: The list of outputs from the model.

        Returns:
            None
        """
        pred_masks: list = []
        iou_predictions: list = []
        pred_labels: list = []
        for output in outputs[0]:
            pred_masks.append(output["masks"][0])
            iou_predictions.append(output["iou_predictions"][0])
            pred_labels.append(output["labels"][0])

        for dataset_item, pred_mask, iou_prediction, labels in zip(
            self.otx_dataset,
            pred_masks,
            iou_predictions,
            pred_labels,
        ):
            annotations: list[Annotation] = []
            for soft_prediction, iou, label in zip(pred_mask, iou_prediction, labels):
                probability = max(min(float(iou), 1.0), 0.0)
                label.probability = probability
                _soft_prediction = soft_prediction.numpy()
                hard_prediction = create_hard_prediction_from_soft_prediction(
                    soft_prediction=_soft_prediction,
                    soft_threshold=0.5,
                )

                if self.use_mask:
                    # set mask as annotation
                    annotation = [
                        Annotation(
                            shape=Image(
                                data=hard_prediction.astype(np.uint8),
                                size=hard_prediction.shape,
                            ),  # type: ignore[arg-type]
                            labels=[ScoredLabel(label=label.label, probability=probability)],
                            id=ID(ObjectId()),
                        ),
                    ]
                else:
                    # generate polygon annotations
                    annotation = create_annotation_from_segmentation_map(
                        hard_prediction=hard_prediction,
                        soft_prediction=_soft_prediction,
                        label_map={1: label.label},
                    )

                annotations.extend(annotation)
            if self.use_mask:
                dataset_item.annotation_scene.append_annotations(annotations)
            else:
                dataset_item.append_annotations(annotations)
