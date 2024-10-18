from anomalib.data import AnomalibDataModule
from anomalib.engine import Engine as AnomalibEngine
from anomalib.models import AnomalyModule

from otx.core.data.module import OTXDataModule
from otx.engine.base import METRICS, Adapter


def wrap_to_anomalib_datamodule(datamodule: OTXDataModule) -> AnomalibDataModule:
    """Mock function to wrap OTXDataModule to AnomalibDataModule."""
    return AnomalibDataModule(
        train=datamodule.train,
        val=datamodule.val,
        test=datamodule.test,
        batch_size=datamodule.batch_size,
        num_workers=datamodule.num_workers,
        pin_memory=datamodule.pin_memory,
        shuffle=datamodule.shuffle,
    )


class AnomalibAdapter(Adapter):
    def __init__(self):
        self._engine = AnomalibEngine()

    def train(
        self,
        model: AnomalyModule,
        datamodule: OTXDataModule | AnomalibDataModule,
        max_epochs: int = 1,
        **kwargs,
    ) -> METRICS:
        if not isinstance(datamodule, AnomalibDataModule):
            datamodule = wrap_to_anomalib_datamodule(datamodule)
        self._engine = AnomalibEngine(max_epochs=max_epochs, **kwargs)
        return self._engine.train(model=model, datamodule=datamodule)

    def test(
        self,
        model: AnomalyModule,
        datamodule: OTXDataModule | AnomalibDataModule,
        max_epochs: int = 1,
        **kwargs,
    ) -> METRICS:
        if not isinstance(datamodule, AnomalibDataModule):
            datamodule = wrap_to_anomalib_datamodule(datamodule)
        self._engine = AnomalibEngine(max_epochs=max_epochs, **kwargs)
        return self._engine.test(model=model, datamodule=datamodule)
