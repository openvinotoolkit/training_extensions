from otx.core.data.entity.detection import DetBatchDataEntity
from otx.core.data.module import OTXDataModule


class TestOTXDataModule:
    def test_train_dataloader(self, fxt_datamodule: OTXDataModule) -> None:
        for batch in fxt_datamodule.train_dataloader():
            assert isinstance(batch, DetBatchDataEntity)
