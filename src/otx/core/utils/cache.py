# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Cache Class for Trainer kwargs."""

from __future__ import annotations

import inspect
import logging
from typing import Any

logger = logging.getLogger(__name__)


class TrainerArgumentsCache:
    """Cache arguments.

    Since the Engine class accepts PyTorch Lightning Trainer arguments, we store these arguments using this class
    before the trainer is instantiated.

    Args:
        (**kwargs): Trainer arguments that are cached

    Example:
        >>> conf = OmegaConf.load("config.yaml")
        >>> cache =  TrainerArgumentsCache(**conf)
        >>> cache.args
        {
            ...
            'max_epochs': 100,
            'val_check_interval': 0
        }
        >>> config = {"max_epochs": 1, "val_check_interval": 1.0}
        >>> cache.update(config)
        Overriding max_epochs from 100 with 1
        Overriding val_check_interval from 0 with 1.0
        >>> cache.args
        {
            ...
            'max_epochs': 1,
            'val_check_interval': 1.0
        }
    """

    def __init__(self, **kwargs) -> None:
        self._cached_args = {**kwargs}
        self.is_trainer_args_identical = False

    def update(self, **kwargs) -> None:
        """Replace cached arguments with arguments retrieved from the model."""
        for key, value in kwargs.items():
            if value is None:
                continue
            if key in self._cached_args and self._cached_args[key] != value:
                logger.info(
                    f"Overriding {key} from {self._cached_args[key]} with {value}",
                )
            self._cached_args[key] = value

    def requires_update(self, **kwargs) -> bool:
        """Checks if the cached arguments need to be updated based on the provided keyword arguments.

        Args:
            **kwargs: The keyword arguments to compare with the cached arguments.

        Returns:
            bool: True if any of the cached arguments need to be updated, False otherwise.
        """
        return not self.is_trainer_args_identical or any(
            key in self._cached_args and self._cached_args[key] != value for key, value in kwargs.items()
        )

    @property
    def args(self) -> dict[str, Any]:
        """Returns the cached arguments.

        Returns:
            dict[str, Any]: The cached arguments.
        """
        return self._cached_args

    @staticmethod
    def get_trainer_constructor_args() -> set[str]:
        """Get the set of arguments accepted by the Trainer class constructor.

        Returns:
            set[str]: A set of argument names accepted by the Trainer class constructor.
        """
        from lightning import Trainer

        sig = inspect.signature(Trainer.__init__)
        return set(sig.parameters.keys())
