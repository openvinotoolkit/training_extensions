# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""OTX export-related types definition."""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from enum import Enum

from otx.core.config.data import TileConfig
from otx.core.types.label import HLabelInfo, LabelInfo


class OTXExportFormatType(str, Enum):
    """OTX export format type definition."""

    ONNX = "ONNX"
    OPENVINO = "OPENVINO"


@dataclass(frozen=True)
class TaskLevelExportParameters:
    """Collection of export parameters which can be defined at a task level.

    Attributes:
        model_type (str): Model type field used in ModelAPI.
        task_type (str): Task type field used in ModelAPI.
        label_info (LabelInfo): OTX label info metadata.
            It will be parsed into a format compatible with ModelAPI.
        optimization_config (dict): Configurations for NNCF PTQ model optimization.
        multilabel (bool | None): Whether it is multilabel or not.
            Only specified for the classification task.
        hierarchical (bool | None): Whether it is hierarchical or not.
            Only specified for the classification task.
        output_raw_scores (bool | None): Whether to output raw scores.
            Only specified for the classification task.
        confidence_threshold (float | None): Confidence threshold for model prediction probability.
            It is used only for classification tasks, detection and instance segmentation tasks.
        iou_threshold (float | None): The Intersection over Union (IoU) threshold
            for Non-Maximum Suppression (NMS) post-processing.
            It is used only for models in detection and instance segmentation tasks.
        return_soft_prediction (bool | None): Whether to return soft prediction.
            It is used only for semantic segmentation tasks.
        soft_threshold (float | None): Minimum class confidence for each pixel.
            The higher the value, the more strict the segmentation is (usually set to 0.5).
            Only specified for semantic segmentation tasks.
        blur_strength (int | None): The higher the value, the smoother the
            segmentation output will be, but less accurate.
            Only specified for semantic segmentation tasks.
        tile_config (TileConfig | None): Configuration for tiling models
            If None, the model is not trained with tiling.
    """

    # Common
    model_type: str
    task_type: str
    label_info: LabelInfo
    optimization_config: dict

    # (Optional) Classification tasks
    multilabel: bool | None = None
    hierarchical: bool | None = None
    output_raw_scores: bool | None = None

    # (Optional) Classification tasks, detection and instance segmentation task
    confidence_threshold: float | None = None

    # (Optional) Detection and instance segmentation task
    iou_threshold: float | None = None

    # (Optional) Semantic segmentation task
    return_soft_prediction: bool | None = None
    soft_threshold: float | None = None
    blur_strength: int | None = None

    # (Optional) Tasks with tiling
    tile_config: TileConfig | None = None

    def wrap(self, **kwargs_to_update) -> TaskLevelExportParameters:
        """Create a new instance by wrapping it with the given keyword arguments.

        Args:
            kwargs_to_update (dict): Keyword arguments to update.

        Returns:
            TaskLevelExportParameters: A new instance with updated attributes.
        """
        updated_kwargs = {field.name: getattr(self, field.name) for field in fields(self)}
        updated_kwargs.update(kwargs_to_update)
        return TaskLevelExportParameters(**updated_kwargs)

    def to_metadata(self) -> dict[tuple[str, str], str]:
        """Convert this dataclass to dictionary format compatible with ModelAPI.

        Returns:
            dict[tuple[str, str], str]: It will be directly delivered to
            OpenVINO IR's `rt_info` or ONNX metadata slot.
        """
        if self.task_type == "instance_segmentation":
            # Instance segmentation needs to add empty label
            all_labels = "otx_empty_lbl "
            all_label_ids = "None "
            for lbl in self.label_info.label_names:
                all_labels += lbl.replace(" ", "_") + " "
                all_label_ids += lbl.replace(" ", "_") + " "
        else:
            all_labels = ""
            all_label_ids = ""
            for lbl in self.label_info.label_names:
                all_labels += lbl.replace(" ", "_") + " "
                all_label_ids += lbl.replace(" ", "_") + " "

        metadata = {
            # Common
            ("model_info", "model_type"): self.model_type,
            ("model_info", "task_type"): self.task_type,
            ("model_info", "label_info"): self.label_info.to_json(),
            ("model_info", "labels"): all_labels.strip(),
            ("model_info", "label_ids"): all_label_ids.strip(),
            ("model_info", "optimization_config"): json.dumps(self.optimization_config),
        }

        if isinstance(self.label_info, HLabelInfo):
            metadata[("model_info", "hierarchical_config")] = json.dumps(
                {
                    "cls_heads_info": self.label_info.as_dict(),
                    "label_tree_edges": self.label_info.label_tree_edges,
                },
            )

        if self.multilabel is not None:
            metadata[("model_info", "multilabel")] = str(self.multilabel)

        if self.hierarchical is not None:
            metadata[("model_info", "hierarchical")] = str(self.hierarchical)

        if self.output_raw_scores is not None:
            metadata[("model_info", "output_raw_scores")] = str(self.output_raw_scores)

        if self.confidence_threshold is not None:
            metadata[("model_info", "confidence_threshold")] = str(self.confidence_threshold)

        if self.iou_threshold is not None:
            metadata[("model_info", "iou_threshold")] = str(self.iou_threshold)

        if self.return_soft_prediction is not None:
            metadata[("model_info", "return_soft_prediction")] = str(self.return_soft_prediction)

        if self.soft_threshold is not None:
            metadata[("model_info", "soft_threshold")] = str(self.soft_threshold)

        if self.blur_strength is not None:
            metadata[("model_info", "blur_strength")] = str(self.blur_strength)

        if self.tile_config is not None:
            metadata.update(
                {
                    ("model_info", "tile_size"): str(self.tile_config.tile_size[0]),
                    ("model_info", "tiles_overlap"): str(self.tile_config.overlap),
                    ("model_info", "max_pred_number"): str(self.tile_config.max_num_instances),
                    ("model_info", "tile_with_full_img"): str(self.tile_config.with_full_img),
                },
            )

        return metadata
