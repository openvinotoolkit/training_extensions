"""
Inference Callbacks for OTE inference
"""

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
from ote_anomalib.data import LabelNames
from ote_anomalib.logging import get_logger
from ote_sdk.entities.datasets import DatasetEntity
from ote_sdk.entities.label import LabelEntity
from ote_sdk.entities.model_template import TaskType
from ote_sdk.entities.result_media import ResultMediaEntity
from ote_sdk.entities.scored_label import ScoredLabel
from ote_sdk.utils.anomaly_utils import create_detection_annotation_from_anomaly_heatmap
from ote_sdk.utils.segmentation_utils import create_annotation_from_segmentation_map
from pytorch_lightning.callbacks import Callback

logger = get_logger(__name__)


class AnomalyInferenceCallback(Callback):
    """Callback that updates the OTE dataset during inference."""

    def __init__(self, ote_dataset: DatasetEntity, labels: List[LabelEntity], task_type: TaskType):
        self.ote_dataset = ote_dataset
        self.normal_label = [label for label in labels if label.name == LabelNames.normal][0]
        self.anomalous_label = [label for label in labels if label.name == LabelNames.anomalous][0]
        self.task_type = task_type
        self.label_map = {0: self.normal_label, 1: self.anomalous_label}

    def on_predict_epoch_end(self, _trainer: pl.Trainer, pl_module: AnomalyModule, outputs: List[Any]):
        """Call when the predict epoch ends."""
        outputs = outputs[0]
        pred_scores = np.hstack([output["pred_scores"].cpu() for output in outputs])
        pred_labels = np.hstack([output["pred_labels"].cpu() for output in outputs])
        anomaly_maps = np.vstack([output["anomaly_maps"].cpu() for output in outputs])
        pred_masks = np.vstack([output["pred_masks"].cpu() for output in outputs])

        # Loop over dataset again to assign predictions
        for dataset_item, pred_score, pred_label, anomaly_map, pred_mask in zip(
            self.ote_dataset, pred_scores, pred_labels, anomaly_maps, pred_masks
        ):
            probability = pred_score if pred_label else 1 - pred_score
            label = self.anomalous_label if pred_label else self.normal_label
            if self.task_type == TaskType.ANOMALY_CLASSIFICATION:
                dataset_item.append_labels([ScoredLabel(label=label, probability=float(probability))])
            if self.task_type == TaskType.ANOMALY_DETECTION:
                annotations = create_detection_annotation_from_anomaly_heatmap(
                    hard_prediction=pred_mask,
                    soft_prediction=anomaly_map,
                    label_map=self.label_map,
                )
                dataset_item.append_annotations(annotations)
                if len(annotations) == 0:
                    dataset_item.append_labels([ScoredLabel(label=self.normal_label, probability=float(probability))])
                else:
                    dataset_item.append_labels(
                        [ScoredLabel(label=self.anomalous_label, probability=float(probability))]
                    )
            elif self.task_type == TaskType.ANOMALY_SEGMENTATION:
                annotations = create_annotation_from_segmentation_map(
                    hard_prediction=pred_mask.squeeze().astype(np.uint8),
                    soft_prediction=anomaly_map.squeeze(),
                    label_map=self.label_map,
                )
                dataset_item.append_annotations(annotations)
                if len(annotations) == 0:
                    dataset_item.append_labels([ScoredLabel(label=self.normal_label, probability=float(probability))])
                else:
                    dataset_item.append_labels(
                        [ScoredLabel(label=self.anomalous_label, probability=float(probability))]
                    )

            dataset_item.append_metadata_item(
                ResultMediaEntity(
                    name="Anomaly Map",
                    type="anomaly_map",
                    annotation_scene=dataset_item.annotation_scene,
                    numpy=anomaly_map_to_color_map(anomaly_map.squeeze(), normalize=False),
                )
            )
            logger.info(
                "\n\tMin: %.3f, Max: %.3f, Threshold: %.3f, Assigned Label '%s', %.3f",
                pl_module.min_max.min.item(),
                pl_module.min_max.max.item(),
                pl_module.image_threshold.value.item(),
                label.name,
                pred_score,
            )
