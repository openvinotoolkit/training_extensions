"""OTX adapters.torch.anomalib.Registry module."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from importlib import import_module

from anomalib.models import _snake_to_pascal_case

from otx.v2.api.core.registry import BaseRegistry

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


class AnomalibRegistry(BaseRegistry):
    def __init__(self, name="anomalib"):
        super().__init__(name)
        self._initialize()

    def _initialize(self):
        for model_name in model_list:
            module = import_module(f"anomalib.models.{model_name}")
            model = getattr(module, f"{_snake_to_pascal_case(model_name)}Lightning")
            if module is not None:
                self.register_module(name=model_name, module=model)

    def get(self, module_type: str):
        # The module_dict is the highest priority.
        if module_type in self.module_dict:
            return self.module_dict[module_type]
        return None
