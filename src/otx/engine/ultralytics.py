from torch.utils.data import DataLoader
from ultralytics.engine.model import Model

from otx.core.data.module import OTXDataModule
from otx.engine.base import METRICS, BaseEngine


def wrap_to_ultralytics_dataset(datamodule: OTXDataModule) -> DataLoader:
    """Mock function to wrap OTXDataModule to ultralytics classification.

    Ideally we want a general ultralytics dataset
    """
    return DataLoader()


class UltralyticsEngine(BaseEngine):
    BASE_MODEL = Model

    def __init__(self, model: Model, **kwargs):
        self.model = model

    @classmethod
    def is_valid_model(cls, model: Model) -> bool:
        return isinstance(model, Model)

    def train(
        self,
        datamodule: OTXDataModule | DataLoader,
        max_epochs: int = 5,
        **kwargs,
    ) -> METRICS:
        if not isinstance(datamodule, DataLoader):
            datamodule = wrap_to_ultralytics_dataset(datamodule)
        print("Pseudo training...")
        return {}  # Metric

    def test(self, datamodule: OTXDataModule | DataLoader, **kwargs) -> METRICS:
        if not isinstance(datamodule, DataLoader):
            datamodule = wrap_to_ultralytics_dataset(datamodule)
        print("Pseudo testing...")
        return {}  # Metric
