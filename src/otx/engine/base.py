from abc import ABC, abstractmethod
from typing import Any

from torch import nn

METRICS = dict[str, float]
ANNOTATIONS = Any


class BaseEngine(ABC):
    BASE_MODEL: nn.Module  # Use this to register models to the CLI

    @classmethod
    @abstractmethod
    def is_valid_model(cls, model: nn.Module) -> bool:
        pass

    @abstractmethod
    def train(self, model: nn.Module, **kwargs) -> METRICS:
        pass

    @abstractmethod
    def test(self, **kwargs) -> METRICS:
        pass

    # @abstractmethod
    # def predict(self, **kwargs) -> ANNOTATIONS:
    #     pass

    # @abstractmethod
    # def export(self, **kwargs) -> Path:
    #     pass

    # @abstractmethod
    # def optimize(self, **kwargs) -> Path:
    #     pass

    # @abstractmethod
    # def explain(self, **kwargs) -> list[Tensor]:
    #     pass

    # @abstractmethod
    # @classmethod
    # def from_config(cls, **kwargs) -> "Backend":
    #     pass

    # @abstractmethod
    # @classmethod
    # def from_model_name(cls, **kwargs) -> "Backend":
    #     pass
