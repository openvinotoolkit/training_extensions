# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Bring from mmpretrain util functions."""
from __future__ import annotations


def get_classification_layers(
    sample_model_dict: dict,
    incremental_model_dict: dict,
    prefix: str = "model.",
) -> dict[str, dict[str, int]]:
    """Get final classification layer information for incremental learning case."""
    classification_layers = {}
    for key in sample_model_dict:
        if sample_model_dict[key].shape != incremental_model_dict[key].shape:
            sample_model_dim = sample_model_dict[key].shape[0]
            incremental_model_dim = incremental_model_dict[key].shape[0]
            stride = incremental_model_dim - sample_model_dim
            num_extra_classes = 6 * sample_model_dim - 5 * incremental_model_dim
            classification_layers[prefix + key] = {"stride": stride, "num_extra_classes": num_extra_classes}
    return classification_layers
