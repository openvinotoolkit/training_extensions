"""Collection of utils for task implementation in Segmentation Task."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from .metadata import get_seg_model_api_configuration
from .processing import get_activation_map

__all__ = ["get_activation_map", "get_seg_model_api_configuration"]
