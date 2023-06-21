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

import pytorch_lightning as pl
import numpy as np
from anomalib.models import AnomalyModule
from pytorch_lightning.callbacks import Callback

from otx.api.entities.annotation import Annotation
from otx.api.entities.image import Image
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.scored_label import ScoredLabel
from otx.api.utils.segmentation_utils import (
    create_annotation_from_segmentation_map,
    create_hard_prediction_from_soft_prediction,
)
from otx.api.entities.id import ID
from bson import ObjectId


class InferenceCallback(Callback):
    """Callback that updates the OTX dataset during inference.
    
    Args:
        otx_dataset (DatasetEntity): 
    """

    def __init__(self, otx_dataset: DatasetEntity):
        self.otx_dataset = otx_dataset.with_empty_annotations()

    def on_predict_epoch_end(self, _trainer: pl.Trainer, _pl_module: AnomalyModule, outputs: List[Any]):
        """Call when the predict epoch ends."""
        outputs = outputs[0]

        # collect generic predictions
        pred_masks = [output["masks"][0] for output in outputs]
        iou_predictions = [output["iou_predictions"][0] for output in outputs]
        gt_labels = [output["labels"][0] for output in outputs]
        for dataset_item, pred_mask, iou_prediction, labels in zip(self.otx_dataset, pred_masks, iou_predictions, gt_labels):
            annotations: List[Annotation] = []
            for soft_prediction, iou, label in zip(pred_mask, iou_prediction, labels):
                probability = max(min(float(iou), 1.), 0.)
                label.probability = probability
                soft_prediction = soft_prediction.numpy()
                hard_prediction = create_hard_prediction_from_soft_prediction(
                    soft_prediction=soft_prediction,
                    soft_threshold=0.5
                )

                if _pl_module.config.dataset.use_mask:
                    # set mask as annotation
                    annotation = [Annotation(
                        shape=Image(data=hard_prediction.astype(np.uint8), size=hard_prediction.shape),
                        labels=[ScoredLabel(label=label.label, probability=probability)],
                        id=ID(ObjectId()),
                    )]
                else:
                    # generate polygon annotations
                    annotation = create_annotation_from_segmentation_map(
                        hard_prediction=hard_prediction,
                        soft_prediction=soft_prediction,
                        label_map={1: label.label},
                    )

                annotations.extend(annotation)
            if _pl_module.config.dataset.use_mask:
                dataset_item.annotation_scene.append_annotations(annotations)
            else:
                dataset_item.append_annotations(annotations)
