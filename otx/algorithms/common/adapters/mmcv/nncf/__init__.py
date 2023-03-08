"""NNCF for mmcv."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# flake8: noqa

from .utils import get_fake_input, model_eval, wrap_nncf_model

__all__ = [
    "get_fake_input",
    "model_eval",
    "wrap_nncf_model",
]
