import logging

from anomalib.data import AnomalibDataModule
from anomalib.models import AnomalyModule
from ultralytics.data import ClassificationDataset as UltralyticsDataset
from ultralytics.engine.model import Model

from otx.core.data.module import OTXDataModule
from otx.core.model.base import OTXModel
from otx.core.utils.cache import TrainerArgumentsCache
from otx.engine.base import METRICS, Adapter

from .anomalib import AnomalibAdapter
from .lightning import LightningAdapter
from .ultralytics import UltralyticsAdapter

logger = logging.getLogger(__name__)


class Engine:
    """Automatically selects the engine based on the model passed to the engine."""

    def __init__(
        self,
        **kwargs,
    ) -> None:
        self._adapter: Adapter | None = None
        self._cache = TrainerArgumentsCache(**kwargs)

    @property
    def adapter(self) -> Adapter:
        return self._adapter

    @adapter.setter
    def adapter(self, adapter: adapter) -> None:
        self._adapter = adapter

    def get_adapter(self, model: OTXModel | AnomalyModule | Model) -> adapter:
        if isinstance(model, AnomalyModule) and (
            self.adapter is not isinstance(self.adapter, AnomalibAdapter) or self.adapter is None
        ):
            self.adapter = AnomalibAdapter(**self._cache.args)
        elif isinstance(model, OTXModel) and (self.adapter is None or not isinstance(self.adapter, LightningAdapter)):
            self.adapter = LightningAdapter(**self._cache.args)
        elif isinstance(model, Model) and (self.adapter is None or not isinstance(self.adapter, UltralyticsAdapter)):
            self.adapter = UltralyticsAdapter(**self._cache.args)

        return self.adapter

    def train(
        self,
        model: OTXModel | AnomalyModule | Model,
        datamodule: OTXDataModule | AnomalibDataModule | UltralyticsDataset,
        **kwargs,
    ) -> METRICS:
        """Train the model."""
        adapter: Adapter = self.get_adapter(model)
        adapter.train(model=model, datamodule=datamodule, **kwargs)

    def test(
        self,
        model: OTXModel | AnomalyModule | Model,
        datamodule: OTXDataModule | AnomalibDataModule | UltralyticsDataset,
    ) -> METRICS:
        """Test the model."""
        adapter = self.get_adapter(model)
        return adapter.test(model=model, datamodule=datamodule)
