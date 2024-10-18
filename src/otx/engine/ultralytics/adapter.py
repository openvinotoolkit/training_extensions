from torch.utils.data import DataLoader
from ultralytics.engine.model import Model

from otx.core.data.module import OTXDataModule
from otx.engine.base import METRICS, Adapter

from .trainer import UltralyticsTrainer


def wrap_to_ultralytics_dataset(datamodule: OTXDataModule) -> DataLoader:
    """Mock function to wrap OTXDataModule to ultralytics classification.

    Ideally we want a general ultralytics dataset
    """
    return DataLoader()


class UltralyticsAdapter(Adapter):
    def __init__(self):
        self._engine: UltralyticsTrainer

    def train(
        self,
        model: Model,
        datamodule: OTXDataModule | DataLoader,
        max_epochs: int = 5,
        **kwargs,
    ) -> METRICS:
        if not isinstance(datamodule, DataLoader):
            datamodule = wrap_to_ultralytics_dataset(datamodule)
        self._engine = UltralyticsTrainer(
            model=model,
            dataloader=datamodule,
            overrides={"epochs": max_epochs, **kwargs},
        )
        return self._engine.train()

    def test(self, model: Model, datamodule: OTXDataModule | DataLoader, **kwargs) -> METRICS:
        if not isinstance(datamodule, DataLoader):
            datamodule = wrap_to_ultralytics_dataset(datamodule)
        self._engine = UltralyticsTrainer(model=model, dataloader=datamodule, overrides={**kwargs})
        return self._engine.train()
