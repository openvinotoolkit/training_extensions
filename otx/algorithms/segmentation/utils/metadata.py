"""Utils for hadnling metadata of segmentation models."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from mmcv.utils import ConfigDict

from otx.api.entities.label_schema import LabelSchemaEntity


def get_seg_model_api_configuration(label_schema: LabelSchemaEntity, hyperparams: ConfigDict):
    """Get ModelAPI config."""
    all_labels = ""
    for lbl in label_schema.get_labels(include_empty=False):
        all_labels += lbl.name.replace(" ", "_") + " "
    all_labels = all_labels.strip()

    return {
        ("model_info", "model_type"): "Segmentation",
        ("model_info", "soft_threshold"): str(hyperparams.postprocessing.soft_threshold),
        ("model_info", "blur_strength"): str(hyperparams.postprocessing.blur_strength),
        ("model_info", "labels"): all_labels,
    }
