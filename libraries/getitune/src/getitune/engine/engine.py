# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Engine base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from pathlib import Path

    from getitune.types import PathLike
    from getitune.types.types import ANNOTATIONS, DATA, METRICS, MODEL


class Engine(ABC):
    """Engine base class."""

    backend_name: ClassVar[str]
    model_base_class: ClassVar[type]

    @abstractmethod
    def train(self, **kwargs) -> METRICS:
        """Train the model."""
        raise NotImplementedError

    @abstractmethod
    def test(self, **kwargs) -> METRICS:
        """Test the model."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, **kwargs) -> ANNOTATIONS:
        """Predict on model."""
        raise NotImplementedError

    @abstractmethod
    def export(self, **kwargs) -> Path:
        """Export the model."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def is_supported(model: MODEL, data: DATA) -> bool:
        """Check if the engine is supported for the given model and data."""
        raise NotImplementedError

    @classmethod
    def from_config(
        cls,
        config_path: PathLike,
        data: DATA,
        work_dir: PathLike | None = None,
        device: str | None = None,
        checkpoint: str | None = None,
        task: str | None = None,
        **kwargs,
    ) -> Engine:
        """Build an engine from a recipe configuration file.

        Args:
            config_path: Path to the recipe YAML file.
            data: DataModule or root directory for the data.
            work_dir: Working directory for the engine. Defaults to None.
            device: Device to use (e.g., ``"auto"``, ``"xpu"``, ``"cpu"``, ``"gpu"``). Defaults to None.
            checkpoint: Optional path to a checkpoint for pretrained or warm-start weights. Defaults to None.
            task: Task type for disambiguation when a model name matches recipes under multiple tasks. Defaults to None.
            **kwargs: Backend-specific keyword arguments forwarded to the engine constructor.

        Returns:
            An instance of the Engine subclass.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        msg = (
            f"{cls.__name__} does not support construction from a recipe config. "
            "Pass a model instance directly to create_engine() instead."
        )
        raise NotImplementedError(msg)

    @property
    @abstractmethod
    def work_dir(self) -> PathLike:
        """Get the working directory for the engine."""
        raise NotImplementedError

    @property
    @abstractmethod
    def model(self) -> MODEL:
        """Returns the model object associated with the engine.

        Returns:
            MODEL: model object.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def datamodule(self) -> DATA:
        """Returns the datamodule object associated with the engine.

        Returns:
            DATA: datamodule object.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def best_checkpoint(self) -> Path | None:
        """Path to the best model checkpoint after training.

        Returns ``None`` if training has not been run or no checkpoint was
        produced.  Subclasses should override to expose their
        backend-specific checkpoint resolution.
        """
        msg = "Subclasses must implement best_checkpoint"
        raise NotImplementedError(msg)
