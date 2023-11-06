"""OTX adapters.torch.lightning.anomalib.Registry module."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from importlib import import_module

from anomalib.models import _snake_to_pascal_case

from otx.v2.adapters.torch.lightning.registry import LightningRegistry

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


class AnomalibRegistry(LightningRegistry):
    """A registry for registering and retrieving anomaly modules.

    Attributes:
        name (str): The name of the registry.
    """

    def __init__(self, name: str = "anomalib") -> None:
        """Initialize a new instance of the AnomalibRegistry class.

        Args:
            name (str): The name of the registry. Defaults to "anomalib".
        """
        super().__init__(name)
        self._initialize()

    def _initialize(self) -> None:
        for model_name in model_list:
            module = import_module(f"anomalib.models.{model_name}")
            model = getattr(module, f"{_snake_to_pascal_case(model_name)}Lightning")
            if module is not None:
                self.register_module(name=model_name, module=model)
