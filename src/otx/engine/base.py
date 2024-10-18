from abc import ABC, abstractmethod
from typing import Any

METRICS = dict[str, float]
ANNOTATIONS = Any


class Adapter(ABC):
    @abstractmethod
    def train(self, **kwargs) -> METRICS:
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
