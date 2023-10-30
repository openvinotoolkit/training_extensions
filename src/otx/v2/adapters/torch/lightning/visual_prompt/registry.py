"""OTX adapters.torch.lightning.visual_prompt.Registry module."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from otx.v2.adapters.torch.lightning.registry import LightningRegistry


class VisualPromptRegistry(LightningRegistry):
    """A registry for registering and retrieving visual_prompt modules.

    Attributes:
        name (str): The name of the registry.
    """

    def __init__(self, name: str = "visual_prompt") -> None:
        """Initialize a new instance of the AnomalibRegistry class.

        Args:
            name (str): The name of the registry. Defaults to "visual_prompt".
        """
        super().__init__(name)
