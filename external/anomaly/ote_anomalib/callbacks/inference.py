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
from anomalib.core.model.anomaly_module import AnomalyModule
from anomalib.utils.post_process import anomaly_map_to_color_map
from ote_anomalib.data import LabelNames
from ote_sdk.entities.annotation import Annotation
from ote_sdk.entities.datasets import DatasetEntity
from ote_sdk.entities.label import LabelEntity
from ote_sdk.entities.result_media import ResultMediaEntity
from ote_sdk.entities.scored_label import ScoredLabel
from ote_sdk.entities.shapes.rectangle import Rectangle
from pytorch_lightning.callbacks import Callback


class InferenceCallback(Callback):
    """
    Callback that updates the OTE dataset during inference.
    """

    def __init__(self, ote_dataset: DatasetEntity, labels: List[LabelEntity]):
        self.ote_dataset = ote_dataset
        self.normal_label = [label for label in labels if label.name == LabelNames.normal][0]
        self.anomalous_label = [label for label in labels if label.name == LabelNames.anomalous][0]

    def on_predict_epoch_end(self, _trainer: pl.Trainer, _pl_module: AnomalyModule, outputs: List[Any]):
        """Called when the predict epoch ends."""
        outputs = outputs[0]
        pred_scores = np.hstack([output["pred_scores"].cpu() for output in outputs])
        pred_labels = np.hstack([output["pred_labels"].cpu() for output in outputs])
        anomaly_maps = np.vstack([output["anomaly_maps"].cpu() for output in outputs])

        # Loop over dataset again to assign predictions
        for dataset_item, pred_score, pred_label, anomaly_map in zip(
            self.ote_dataset, pred_scores, pred_labels, anomaly_maps
        ):

            assigned_label = self.anomalous_label if pred_label else self.normal_label
            shape = Annotation(
                Rectangle(x1=0, y1=0, x2=1, y2=1),
                labels=[ScoredLabel(assigned_label, probability=pred_score)],
            )

            dataset_item.append_annotations([shape])

            heatmap = anomaly_map_to_color_map(anomaly_map.squeeze())
            heatmap_media = ResultMediaEntity(
                name="Anomaly Map", type="anomaly_map", annotation_scene=dataset_item.annotation_scene, numpy=heatmap
            )
            dataset_item.append_metadata_item(heatmap_media)
