import logging
from pathlib import Path

from torch import nn

from otx.engine.base import BaseEngine

from .base import BaseEngine

logger = logging.getLogger(__name__)


class AutoConfigurator:
    """Mock autoconfigurator for the engine."""

    def __init__(self, model: nn.Module | None = None, data_root: Path | None = None, task: str | None = None):
        self._engine = self._configure_engine(model)  # ideally we want to pass the data_root and task as well

    @property
    def engine(self) -> BaseEngine:
        return self._engine

    def _configure_engine(self, model: nn.Module) -> BaseEngine:
        for engine in BaseEngine.__subclasses__():
            if engine.is_valid_model(model):
                logger.info(f"Using {engine.__name__} for model {model.__class__.__name__}")
                return engine(model=model)
        raise ValueError(f"Model {model} is not supported by any of the engines.")


class Engine:
    """Automatically selects the engine based on the model passed to the engine."""

    def __new__(
        cls,
        model: nn.Module,
        data_root: Path | None = None,
        **kwargs,
    ) -> BaseEngine:
        """This takes in all the parameters that are currently passed to the OTX Engine's `__init__` method."""
        autoconfigurator = AutoConfigurator(model, data_root=data_root, **kwargs)
        return autoconfigurator.engine
