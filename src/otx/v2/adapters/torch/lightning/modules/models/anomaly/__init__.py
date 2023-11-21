"""Setting for OTX Anomalib Models."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from importlib import import_module

from anomalib.models import _snake_to_pascal_case

from otx.v2.adapters.torch.lightning.modules.models import MODELS

# [TODO]: This should change with anomalib v1.0.
# COPY from anomalib.models.__init__.py
model_list = [
    "cfa",
    "cflow",
    "csflow",
    "dfkde",
    "dfm",
    "draem",
    "fastflow",
    "ganomaly",
    "padim",
    "patchcore",
    "reverse_distillation",
    "rkde",
    "stfpm",
    "ai_vad",
]


# NOTE: Register the model with the Registry to make it available via the config API.
for model_name in model_list:
    module = import_module(f"anomalib.models.{model_name}")
    model = getattr(module, f"{_snake_to_pascal_case(model_name)}Lightning")
    if module is not None:
        MODELS.register_module(name=model_name, module=model)
