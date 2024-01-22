# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Utility functions."""

from __future__ import annotations

from collections import OrderedDict


def remove_model_form_weight_key(weight: dict) -> dict:
    """Remove all 'model.' in keys in the model weight."""
    new_weight = OrderedDict()
    weight_to_fix = weight["state_dict"] if "state_dict" in weight else weight

    for key, val in weight_to_fix.items():
        new_key = key
        while "model." in new_key:
            new_key = new_key.replace("model.", "")
        new_weight[new_key] = val

    if "state_dict" in weight:
        weight["state_dict"] = new_weight
        return weight

    return new_weight
