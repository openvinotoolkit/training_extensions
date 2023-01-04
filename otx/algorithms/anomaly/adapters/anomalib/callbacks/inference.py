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
from anomalib.models import AnomalyModule
from anomalib.post_processing import anomaly_map_to_color_map
from pytorch_lightning.callbacks import Callback

from otx.api.entities.shapes.rectangle import Rectangle
from otx.api.entities.annotation import Annotation
from otx.algorithms.anomaly.adapters.anomalib.logger import get_logger
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.label import LabelEntity
from otx.api.entities.model_template import TaskType
from otx.api.entities.result_media import ResultMediaEntity
from otx.api.entities.scored_label import ScoredLabel
from otx.api.utils.anomaly_utils import create_detection_annotation_from_anomaly_heatmap
from otx.api.utils.segmentation_utils import create_annotation_from_segmentation_map

logger = get_logger(__name__)


class AnomalyInferenceCallback(Callback):
    """Callback that updates the OTX dataset during inference."""

    def __init__(self, otx_dataset: DatasetEntity, labels: List[LabelEntity], task_type: TaskType):
        self.otx_dataset = otx_dataset
        self.normal_label = [label for label in labels if not label.is_anomalous][0]
        self.anomalous_label = [label for label in labels if label.is_anomalous][0]
        self.task_type = task_type
        self.label_map = {0: self.normal_label, 1: self.anomalous_label}

    def on_predict_epoch_end(self, _trainer: pl.Trainer, pl_module: AnomalyModule, outputs: List[Any]):
        """Call when the predict epoch ends."""
        # TODO; refactor Ignore too many locals
        # pylint: disable=too-many-locals
        outputs = outputs[0]
        pred_scores = np.hstack([output["pred_scores"].cpu() for output in outputs])
        pred_labels = np.hstack([output["pred_labels"].cpu() for output in outputs])
        anomaly_maps = np.vstack([output["anomaly_maps"].cpu() for output in outputs])
        pred_masks = np.vstack([output["pred_masks"].cpu() for output in outputs])

        if self.task_type == TaskType.ACTION_CLASSIFICATION:
            self._process_classification_predictions(pred_labels, pred_scores)

        if self.task_type == TaskType.ANOMALY_DETECTION:
            pred_boxes = []
            box_scores = []
            box_labels = []
            [pred_boxes.extend(output["pred_boxes"]) for output in outputs]
            [box_scores.extend(output["box_scores"]) for output in outputs]
            [box_labels.extend(output["box_labels"]) for output in outputs]
            
            self._process_detection_predictions(pred_boxes, box_scores, box_labels, pred_scores, pred_masks.shape[-2:])

        elif self.task_type == TaskType.ANOMALY_SEGMENTATION:
            self._process_segmentation_predictions(pred_masks, anomaly_maps, pred_scores)

        # add anomaly map as metadata
        for dataset_item, anomaly_map in zip(self.otx_dataset, anomaly_maps):
            dataset_item.append_metadata_item(
                ResultMediaEntity(
                    name="Anomaly Map",
                    type="anomaly_map",
                    annotation_scene=dataset_item.annotation_scene,
                    numpy=anomaly_map_to_color_map(anomaly_map.squeeze(), normalize=False),
                )
            )

    def _process_classification_predictions(self, pred_labels, pred_scores):
        for dataset_item, pred_label, pred_score in zip(self.otx_dataset, pred_labels, pred_scores):
            # get label
            label = self.anomalous_label if pred_label else self.normal_label
            probability = pred_score if pred_label else 1 - pred_score
            # update dataset item
            dataset_item.append_labels([ScoredLabel(label=label, probability=float(probability))])

    def _process_detection_predictions(self, pred_boxes, box_scores, box_labels, pred_scores, image_size):

        height, width = image_size
        for dataset_item, im_boxes, im_box_scores, im_box_labels, pred_score in zip(self.otx_dataset, pred_boxes, box_scores, box_labels, pred_scores):
            # generate annotations
            annotations: List[Annotation] = []
            for box, score, label in zip(im_boxes, im_box_scores, im_box_labels):
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
            label = self.normal_label if len(annotations) == 0 else self.anomalous_label
            probability = pred_score if label.is_anomalous else 1 - pred_score
            # update dataset item
            dataset_item.append_annotations(annotations)
            dataset_item.append_labels([ScoredLabel(label=label, probability=float(probability))])

    def _process_segmentation_predictions(self, pred_masks, anomaly_maps, pred_scores):

        for dataset_item, pred_mask, anomaly_map, pred_score in zip(self.otx_dataset, pred_masks, anomaly_maps, pred_scores):
            # generate polygon annotations
            annotations = create_annotation_from_segmentation_map(
                hard_prediction=pred_mask.squeeze().astype(np.uint8),
                soft_prediction=anomaly_map.squeeze(),
                label_map=self.label_map,
            )
            # get label
            label = self.normal_label if len(annotations) == 0 else self.anomalous_label
            probability = pred_score if label.is_anomalous else 1 - pred_score
            # update dataset item
            dataset_item.append_annotations(annotations)
            dataset_item.append_labels([ScoredLabel(label=label, probability=float(probability))])
