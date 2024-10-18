from anomalib.data import AnomalibDataModule
from anomalib.engine import Engine as AnomalibEngine
from anomalib.models import AnomalyModule

from otx.core.data.module import OTXDataModule
from otx.engine.base import METRICS, BaseEngine


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


class AnomalyEngine(BaseEngine):
    BASE_MODEL = AnomalyModule

    def __init__(self, model: AnomalyModule, **kwargs):
        self.model = model
        self._engine = AnomalibEngine()

    @classmethod
    def is_valid_model(cls, model: AnomalyModule) -> bool:
        return isinstance(model, AnomalyModule)

    def train(
        self,
        datamodule: OTXDataModule | AnomalibDataModule,
        **kwargs,
    ) -> METRICS:
        if not isinstance(datamodule, AnomalibDataModule):
            datamodule = wrap_to_anomalib_datamodule(datamodule)
        print("Pseudo training...")

    def test(
        self,
        datamodule: OTXDataModule | AnomalibDataModule,
        **kwargs,
    ) -> METRICS:
        if not isinstance(datamodule, AnomalibDataModule):
            datamodule = wrap_to_anomalib_datamodule(datamodule)
        print("Pseudo testing...")
