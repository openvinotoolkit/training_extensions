
from lightning import Callback

class MMDetSwitchPipeline(Callback):
    def __init__(self, transforms) -> None:
        super().__init__()
        self.transforms = transforms

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        dm = trainer.datamodule
