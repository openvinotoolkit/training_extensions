"""Inference Callbacks for OTX inference."""

# Copyright (C) 2021 Intel Corporation
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
import pytorch_lightning as pl
import torch
from anomalib.models import AnomalyModule
from pytorch_lightning.callbacks import Callback
from torch import Tensor

from otx.api.entities.annotation import Annotation
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.label import LabelEntity
from otx.api.entities.model_template import TaskType
from otx.api.entities.result_media import ResultMediaEntity
from otx.api.entities.scored_label import ScoredLabel
from otx.api.entities.shapes.rectangle import Rectangle
from otx.api.utils.segmentation_utils import create_annotation_from_segmentation_map
from otx.utils.logger import get_logger

logger = get_logger()


class AnomalyInferenceCallback(Callback):
    """Callback that updates the OTX dataset during inference."""

    def __init__(self, otx_dataset: DatasetEntity, labels: List[LabelEntity], task_type: TaskType):
        self.otx_dataset = otx_dataset
        self.normal_label = [label for label in labels if not label.is_anomalous][0]
        self.anomalous_label = [label for label in labels if label.is_anomalous][0]
        self.task_type = task_type
        self.label_map = {0: self.normal_label, 1: self.anomalous_label}

    def on_predict_epoch_end(self, _trainer: pl.Trainer, _pl_module: AnomalyModule, outputs: List[Any]):
        """Call when the predict epoch ends."""
        # TODO; refactor Ignore too many locals
        # pylint: disable=too-many-locals
        outputs = outputs[0]
        # collect generic predictions
        pred_scores = torch.hstack([output["pred_scores"].cpu() for output in outputs])
        pred_labels = torch.hstack([output["pred_labels"].cpu() for output in outputs])
        anomaly_maps = torch.vstack([output["anomaly_maps"].cpu() for output in outputs])
        pred_masks = torch.vstack([output["pred_masks"].cpu() for output in outputs])

        # add the predictions to the dataset item depending on the task type
        if self.task_type == TaskType.ANOMALY_CLASSIFICATION:
            self._process_classification_predictions(pred_labels, pred_scores)
        elif self.task_type == TaskType.ANOMALY_DETECTION:
            # collect detection predictions
            pred_boxes = []
            box_scores = []
            box_labels = []
            for output in outputs:
                pred_boxes.extend(output["pred_boxes"])
                box_scores.extend(output["box_scores"])
                box_labels.extend(output["box_labels"])

            self._process_detection_predictions(pred_boxes, box_scores, box_labels, pred_scores, pred_masks.shape[-2:])
        elif self.task_type == TaskType.ANOMALY_SEGMENTATION:
            self._process_segmentation_predictions(pred_masks, anomaly_maps, pred_scores)

        # add anomaly map as metadata
        for dataset_item, anomaly_map in zip(self.otx_dataset, anomaly_maps):
            dataset_item.append_metadata_item(
                ResultMediaEntity(
                    name="Anomaly Map",
                    type="anomaly_map",
                    label=dataset_item.annotation_scene.get_labels()[0],
                    annotation_scene=dataset_item.annotation_scene,
                    numpy=(anomaly_map * 255).squeeze().cpu().numpy().astype(np.uint8),
                )
            )

    def _process_classification_predictions(self, pred_labels: Tensor, pred_scores: Tensor):
        """Add classification predictions to the dataset items.

        Args:
            pred_labels (Tensor): Predicted image labels.
            pred_scores (Tensor): Predicted image-level anomaly scores.
        """
        for dataset_item, pred_label, pred_score in zip(self.otx_dataset, pred_labels, pred_scores):
            # get label
            label = self.anomalous_label if pred_label else self.normal_label
            probability = pred_score if pred_label else 1 - pred_score
            # update dataset item
            dataset_item.append_labels([ScoredLabel(label=label, probability=float(probability))])

    def _process_detection_predictions(
        self,
        pred_boxes: List[Tensor],
        box_scores: List[Tensor],
        box_labels: List[Tensor],
        pred_scores: Tensor,
        image_size: torch.Size,
    ):
        """Add detection predictions to the dataset items.

        Args:
            pred_boxes (List[Tensor]): Predicted bounding box locations.
            box_scores (List[Tensor]): Predicted anomaly scores for the bounding boxes.
            box_labels (List[Tensor]): Predicted labels for the bounding boxes.
            pred_scores (Tensor): Predicted image-level anomaly scores.
            image_size: (torch.Size): Image size of the original images.
        """
        # TODO; refactor Ignore too many locals
        # pylint: disable=too-many-locals
        height, width = image_size
        for dataset_item, im_boxes, im_box_scores, im_box_labels, pred_score in zip(
            self.otx_dataset, pred_boxes, box_scores, box_labels, pred_scores
        ):
            # generate annotations
            annotations: List[Annotation] = []
            for box, score, label in zip(im_boxes, im_box_scores, im_box_labels):
                if box[0] >= box[2] or box[1] >= box[3]:  # discard 1-pixel boxes
                    continue
                shape = Rectangle(
                    x1=box[0].item() / width,
                    y1=box[1].item() / height,
                    x2=box[2].item() / width,
                    y2=box[3].item() / height,
                )
                label = self.label_map[label.item()]
                probability = score.item()
                annotations.append(Annotation(shape=shape, labels=[ScoredLabel(label=label, probability=probability)]))
            # get label
            label = self.anomalous_label if annotations else self.normal_label
            probability = pred_score if label.is_anomalous else 1 - pred_score
            # update dataset item
            dataset_item.append_annotations(annotations)
            dataset_item.append_labels([ScoredLabel(label=label, probability=float(probability))])

    def _process_segmentation_predictions(self, pred_masks: Tensor, anomaly_maps: Tensor, pred_scores: Tensor):
        """Add segmentation predictions to the dataset items.

        Args:
            pred_masks (Tensor): Predicted anomaly masks.
            anomaly_maps (Tensor): Predicted pixel-level anomaly scores.
            pred_scores (Tensor): Predicted image-level anomaly scores.
        """
        for dataset_item, pred_mask, anomaly_map, pred_score in zip(
            self.otx_dataset, pred_masks, anomaly_maps, pred_scores
        ):
            # generate polygon annotations
            annotations = create_annotation_from_segmentation_map(
                hard_prediction=pred_mask.squeeze().numpy().astype(np.uint8),
                soft_prediction=anomaly_map.squeeze().numpy(),
                label_map=self.label_map,
            )
            # get label
            label = self.normal_label if len(annotations) == 0 else self.anomalous_label
            probability = pred_score if label.is_anomalous else 1 - pred_score
            # update dataset item
            dataset_item.append_annotations(annotations)
            dataset_item.append_labels([ScoredLabel(label=label, probability=float(probability))])
