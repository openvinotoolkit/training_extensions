"""OTX adapters.torch.lightning.Registry module."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Callable

from otx.v2.api.core.registry import BaseRegistry


class LightningRegistry(BaseRegistry):
    """A registry for registering and retrieving lightning modules.

    Attributes:
        name (str): The name of the registry.
    """

    def __init__(self, name: str = "lightning") -> None:
        """Initialize a new instance of the LightningRegistry class.

        Args:
            name (str): The name of the registry. Defaults to "lightning".
        """
        super().__init__(name)

    def get(self, module_type: str) -> Callable | None:
        """Retrieve a registered module by its type.

        Args:
            module_type (str): The type of the module to retrieve.

        Returns:
            Callable | None: The registered module, or None if not found.
        """
        # The module_dict is the highest priority.
        if module_type in self.module_dict:
            return self.module_dict[module_type]
        return None
