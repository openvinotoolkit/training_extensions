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
from anomalib.models import AnomalyModule
from pytorch_lightning.callbacks import Callback

from otx.api.entities.annotation import Annotation
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.id import ID
from otx.api.entities.label import Domain, LabelEntity
from otx.api.entities.scored_label import ScoredLabel
from otx.api.utils.segmentation_utils import (
    create_annotation_from_segmentation_map,
    create_hard_prediction_from_soft_prediction,
)


class InferenceCallback(Callback):
    """Callback that updates the OTX dataset during inference."""

    def __init__(self, otx_dataset: DatasetEntity):
        self.otx_dataset = otx_dataset.with_empty_annotations()

    def on_predict_epoch_end(self, _trainer: pl.Trainer, _pl_module: AnomalyModule, outputs: List[Any]):
        """Call when the predict epoch ends."""
        outputs = outputs[0]

        # collect generic predictions
        pred_masks = [output['masks'][0] for output in outputs]
        iou_predictions = [output['iou_predictions'][0] for output in outputs]
        label_map = LabelEntity(
            name="foreground",
            domain=Domain.VISUAL_PROMPTING,
            is_empty=False, id=ID(1)
        )
        for dataset_item, pred_mask, iou_prediction in zip(self.otx_dataset, pred_masks, iou_predictions):
            annotations: List[Annotation] = []
            for soft_prediction in pred_mask:
                soft_prediction = soft_prediction.numpy()
                hard_prediction = create_hard_prediction_from_soft_prediction(
                    soft_prediction=soft_prediction,
                    soft_threshold=0.5
                )

                # generate polygon annotations
                annotation = create_annotation_from_segmentation_map(
                    hard_prediction=hard_prediction,
                    soft_prediction=soft_prediction,
                    label_map={1: label_map},
                )
                annotations.extend(annotation)

            dataset_item.append_annotations(annotations)
            dataset_item.append_labels([
                ScoredLabel(label=label_map, probability=max(min(float(iou), 1.), 0.)) for iou in iou_prediction]
            )
