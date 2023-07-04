"""Inference Callbacks for OTX inference."""

# Copyright (C) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

from typing import Any, List

import numpy as np
from bson import ObjectId
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback

from otx.api.entities.annotation import Annotation
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.id import ID
from otx.api.entities.image import Image
from otx.api.entities.scored_label import ScoredLabel
from otx.api.utils.segmentation_utils import (
    create_annotation_from_segmentation_map,
    create_hard_prediction_from_soft_prediction,
)


class InferenceCallback(Callback):
    """Callback that updates otx_dataset during inference.

    Args:
        otx_dataset (DatasetEntity): Dataset that predictions will be updated.
    """

    def __init__(self, otx_dataset: DatasetEntity):
        # decide if using mask or polygon annotations for predictions
        if any(isinstance(shape, Image) for shape in otx_dataset[0].annotation_scene.shapes):
            self.use_mask = True
        else:
            self.use_mask = False
        self.otx_dataset = otx_dataset.with_empty_annotations()

    def on_predict_epoch_end(self, _trainer: Trainer, _pl_module: LightningModule, outputs: List[Any]) -> None:
        """Call when the predict epoch ends."""
        # collect generic predictions
        pred_masks: List = []
        iou_predictions: List = []
        pred_labels: List = []
        for output in outputs[0]:
            pred_masks.append(output["masks"][0])
            iou_predictions.append(output["iou_predictions"][0])
            pred_labels.append(output["labels"][0])

        for dataset_item, pred_mask, iou_prediction, labels in zip(
            self.otx_dataset, pred_masks, iou_predictions, pred_labels
        ):
            annotations: List[Annotation] = []
            for soft_prediction, iou, label in zip(pred_mask, iou_prediction, labels):
                probability = max(min(float(iou), 1.0), 0.0)
                label.probability = probability
                soft_prediction = soft_prediction.numpy()
                hard_prediction = create_hard_prediction_from_soft_prediction(
                    soft_prediction=soft_prediction, soft_threshold=0.5
                )

                if self.use_mask:
                    # set mask as annotation
                    annotation = [
                        Annotation(
                            shape=Image(
                                data=hard_prediction.astype(np.uint8), size=hard_prediction.shape
                            ),  # type: ignore[arg-type]
                            labels=[ScoredLabel(label=label.label, probability=probability)],
                            id=ID(ObjectId()),
                        )
                    ]
                else:
                    # generate polygon annotations
                    annotation = create_annotation_from_segmentation_map(
                        hard_prediction=hard_prediction,
                        soft_prediction=soft_prediction,
                        label_map={1: label.label},
                    )

                annotations.extend(annotation)
            if self.use_mask:
                dataset_item.annotation_scene.append_annotations(annotations)
            else:
                dataset_item.append_annotations(annotations)
