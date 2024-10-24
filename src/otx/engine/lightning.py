import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from lightning.pytorch import Trainer

from otx.core.data.module import OTXDataModule
from otx.core.metrics import MetricCallable
from otx.core.model.base import OTXModel
from otx.core.types.task import OTXTaskType
from otx.core.utils.cache import TrainerArgumentsCache

from .base import Adapter


@contextmanager
def override_metric_callable(model: OTXModel, new_metric_callable: MetricCallable | None) -> Iterator[OTXModel]:
    """Override `OTXModel.metric_callable` to change the evaluation metric.

    Args:
        model: Model to override its metric callable
        new_metric_callable: If not None, override the model's one with this. Otherwise, do not override.
    """
    if new_metric_callable is None:
        yield model
        return

    orig_metric_callable = model.metric_callable
    try:
        model.metric_callable = new_metric_callable
        yield model
    finally:
        model.metric_callable = orig_metric_callable


class LightningAdapter(Adapter):
    """OTX Engine.

    This is a temporary name and we can change it later. It is basically a subset of what is currently present in the
    original OTX Engine class (engine.py)
    """

    def __init__(
        self,
        datamodule: OTXDataModule | None = None,
        model: OTXModel | str | None = None,
        task: OTXTaskType | None = None,
        **kwargs,
    ):
        self._cache = TrainerArgumentsCache(**kwargs)
        self.task = task
        self._trainer: Trainer | None = None
        self._datamodule: OTXDataModule = datamodule
        self._model: OTXModel = model

    def train(
        self,
        model: OTXModel | None = None,
        datamodule: OTXDataModule | None = None,
        max_epochs: int = 10,
        deterministic: bool = True,
        val_check_interval: int | float | None = 1,
        metric: MetricCallable | None = None,
    ) -> dict[str, float]:
        if model is not None:
            self.model = model
        if datamodule is not None:
            self.datamodule = datamodule
        self._build_trainer(
            logger=None,
            callbacks=None,
            max_epochs=max_epochs,
            deterministic=deterministic,
            val_check_interval=val_check_interval,
        )

        # NOTE: Model's label info should be converted datamodule's label info before ckpt loading
        # This is due to smart weight loading check label name as well as number of classes.
        if self.model.label_info != self.datamodule.label_info:
            msg = (
                "Model label_info is not equal to the Datamodule label_info. "
                f"It will be overriden: {self.model.label_info} => {self.datamodule.label_info}"
            )
            logging.warning(msg)
            self.model.label_info = self.datamodule.label_info

        with override_metric_callable(model=self.model, new_metric_callable=metric) as model:
            self.trainer.fit(
                model=model,
                datamodule=self.datamodule,
            )
        self.checkpoint = self.trainer.checkpoint_callback.best_model_path

        if not isinstance(self.checkpoint, (Path, str)):
            msg = "self.checkpoint should be Path or str at this time."
            raise TypeError(msg)

        best_checkpoint_symlink = Path(self.work_dir) / "best_checkpoint.ckpt"
        if best_checkpoint_symlink.is_symlink():
            best_checkpoint_symlink.unlink()
        best_checkpoint_symlink.symlink_to(self.checkpoint)

        return self.trainer.callback_metrics

    def test(self, **kwargs) -> dict[str, float]:
        pass

    @property
    def trainer(self) -> Trainer:
        """Returns the trainer object associated with the engine.

        To get this property, you should execute `Engine.train()` function first.

        Returns:
            Trainer: The trainer object.
        """
        if self._trainer is None:
            msg = "Please run train() first"
            raise RuntimeError(msg)
        return self._trainer

    def _build_trainer(self, **kwargs) -> None:
        """Instantiate the trainer based on the model parameters."""
        if self._cache.requires_update(**kwargs) or self._trainer is None:
            self._cache.update(**kwargs)

            kwargs = self._cache.args
            self._trainer = Trainer(**kwargs)
            self._cache.is_trainer_args_identical = True
            self._trainer.task = self.task
            self.work_dir = self._trainer.default_root_dir

    @property
    def model(self) -> OTXModel:
        return self._model

    @model.setter
    def model(self, model: OTXModel) -> None:
        self._model = model

    @property
    def datamodule(self) -> OTXDataModule:
        return self._datamodule

    @datamodule.setter
    def datamodule(self, datamodule: OTXDataModule) -> None:
        self._datamodule = datamodule
