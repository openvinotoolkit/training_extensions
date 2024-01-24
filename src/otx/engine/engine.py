"""Module for OTX engine components."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable

import torch
from lightning import Trainer, seed_everything

from otx.core.config.device import DeviceConfig
from otx.core.config.explain import ExplainConfig
from otx.core.data.module import OTXDataModule
from otx.core.model.entity.base import OTXModel
from otx.core.model.module.base import OTXLitModule
from otx.core.types.device import DeviceType
from otx.core.types.task import OTXTaskType
from otx.core.utils.cache import TrainerArgumentsCache

if TYPE_CHECKING:
    from pathlib import Path

    from lightning import Callback
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
    from lightning.pytorch.loggers import Logger
    from lightning.pytorch.utilities.types import EVAL_DATALOADERS
    from pytorch_lightning.trainer.connectors.accelerator_connector import _PRECISION_INPUT


LITMODULE_PER_TASK = {
    OTXTaskType.MULTI_CLASS_CLS: "otx.core.model.module.classification.OTXMulticlassClsLitModule",
    OTXTaskType.MULTI_LABEL_CLS: "otx.core.model.module.classification.OTXMultilabelClsLitModule",
    OTXTaskType.H_LABEL_CLS: "otx.core.model.module.classification.OTXHlabelClsLitModule",
    OTXTaskType.DETECTION: "otx.core.model.module.detection.OTXDetectionLitModule",
    OTXTaskType.INSTANCE_SEGMENTATION: "otx.core.model.module.instance_segmentation.OTXInstanceSegLitModule",
    OTXTaskType.SEMANTIC_SEGMENTATION: "otx.core.model.module.segmentation.OTXSegmentationLitModule",
    OTXTaskType.ACTION_CLASSIFICATION: "otx.core.model.module.action_classification.OTXActionClsLitModule",
    OTXTaskType.ACTION_DETECTION: "otx.core.model.module.action_detection.OTXActionDetLitModule",
    OTXTaskType.VISUAL_PROMPTING: "otx.core.model.module.visual_prompting.OTXVisualPromptingLitModule",
}


class Engine:
    """OTX Engine.

    This class defines the Engine for OTX, which governs each step of the OTX workflow.

    Example:
        The following examples show how to use the Engine class.

        Auto-Configuration with data_root
        >>> engine = Engine(
        ...     data_root=<dataset/path>,
        ... )

        Create Engine with Custom OTXModel
        >>> engine = Engine(
        ...     data_root=<dataset/path>,
        ...     model=OTXModel(...),
        ...     checkpoint=<checkpoint/path>,
        ... )

        Create Engine with Custom OTXDataModule
        >>> engine = Engine(
        ...     model = OTXModel(...),
        ...     datamodule = OTXDataModule(...),
        ... )
    """

    def __init__(
        self,
        *,
        data_root: str | Path | None = None,
        task: OTXTaskType | None = None,
        work_dir: str | Path = "./otx-workspace",
        datamodule: OTXDataModule | None = None,
        model: OTXModel | str | None = None,
        optimizer: OptimizerCallable | None = None,
        scheduler: LRSchedulerCallable | None = None,
        checkpoint: str | None = None,
        device: DeviceType = DeviceType.auto,
        **kwargs,
    ):
        """Initializes the OTX Engine.

        Args:
            data_root (str | Path | None, optional): Root directory for the data. Defaults to None.
            task (OTXTaskType | None, optional): The type of OTX task. Defaults to None.
            work_dir (str | Path, optional): Working directory for the engine. Defaults to "./otx-workspace".
            datamodule (OTXDataModule | None, optional): The data module for the engine. Defaults to None.
            model (OTXModel | str | None, optional): The model for the engine. Defaults to None.
            optimizer (OptimizerCallable | None, optional): The optimizer for the engine. Defaults to None.
            scheduler (LRSchedulerCallable | None, optional): The learning rate scheduler for the engine.
                Defaults to None.
            checkpoint (str | None, optional): Path to the checkpoint file. Defaults to None.
            device (DeviceType, optional): The device type to use. Defaults to DeviceType.auto.
            **kwargs: Additional keyword arguments for pl.Trainer.
        """
        self.work_dir = work_dir
        self.checkpoint = checkpoint
        self.device = DeviceConfig(accelerator=device)
        self._cache = TrainerArgumentsCache(
            default_root_dir=self.work_dir,
            accelerator=self.device.accelerator,
            devices=self.device.devices,
            **kwargs,
        )

        # [TODO] harimkang: It will be updated in next PR.
        if not isinstance(model, OTXModel) or datamodule is None or optimizer is None or scheduler is None:
            msg = "Auto-Configuration is not implemented yet."
            raise NotImplementedError(msg)
        self.datamodule: OTXDataModule = datamodule
        self.task = self.datamodule.task

        self._trainer: Trainer | None = None
        self._model: OTXModel = model
        self.optimizer: OptimizerCallable = optimizer
        self.scheduler: LRSchedulerCallable = scheduler

    # ------------------------------------------------------------------------ #
    # General OTX Entry Points
    # ------------------------------------------------------------------------ #

    def train(
        self,
        max_epochs: int = 10,
        seed: int | None = None,
        deterministic: bool = False,
        precision: _PRECISION_INPUT | None = "32",
        val_check_interval: int | float | None = None,
        callbacks: list[Callback] | Callback | None = None,
        logger: Logger | Iterable[Logger] | bool | None = None,
        resume: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        """Trains the model using the provided LightningModule and OTXDataModule.

        Args:
            max_epochs (int | None, optional): The maximum number of epochs. Defaults to None.
            seed (int | None, optional): The random seed. Defaults to None.
            deterministic (bool | None, optional): Whether to enable deterministic behavior. Defaults to False.
            precision (_PRECISION_INPUT | None, optional): The precision of the model. Defaults to 32.
            val_check_interval (int | float | None, optional): The validation check interval. Defaults to None.
            callbacks (list[Callback] | Callback | None, optional): The callbacks to be used during training.
            logger (Logger | Iterable[Logger] | bool | None, optional): The logger(s) to be used. Defaults to None.
            resume (bool, optional): If True, tries to resume training from existing checkpoint.
            **kwargs: Additional keyword arguments for pl.Trainer configuration.

        Returns:
            dict[str, Any]: A dictionary containing the callback metrics from the trainer.

        Example:
            >>> engine.train(
            ...     max_epochs=3,
            ...     seed=1234,
            ...     deterministic=False,
            ...     precision="32",
            ... )

        CLI Usage:
            1. you can train with data_root only. then OTX will provide default model.
                ```python
                otx train --data_root <DATASET_PATH>
                ```
            2. you can pick a model or datamodule as Config file or Class.
                ```python
                otx train
                --data_root <DATASET_PATH>
                --model <CONFIG | CLASS_PATH_OR_NAME> --data <CONFIG | CLASS_PATH_OR_NAME>
                ```
            3. Of course, you can override the various values with commands.
                ```python
                otx train
                    --data_root <DATASET_PATH>
                    --max_epochs <EPOCHS, int> --checkpoint <CKPT_PATH, str>
                ```
            4. If you have a complete configuration file, run it like this.
                ```python
                otx train --data_root <DATASET_PATH> --config <CONFIG_PATH, str>
                ```
        """
        lit_module = self._build_lightning_module(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
        )
        lit_module.meta_info = self.datamodule.meta_info

        if seed is not None:
            seed_everything(seed, workers=True)

        self._build_trainer(
            logger=logger,
            callbacks=callbacks,
            precision=precision,
            max_epochs=max_epochs,
            deterministic=deterministic,
            val_check_interval=val_check_interval,
            **kwargs,
        )
        fit_kwargs: dict[str, Any] = {}
        if resume:
            fit_kwargs["ckpt_path"] = self.checkpoint
        elif self.checkpoint is not None:
            loaded_checkpoint = torch.load(self.checkpoint)
            # loaded checkpoint have keys (OTX1.5): model, config, labels, input_size, VERSION
            lit_module.load_state_dict(loaded_checkpoint)

        self.trainer.fit(
            model=lit_module,
            datamodule=self.datamodule,
            **fit_kwargs,
        )
        self.checkpoint = self.trainer.checkpoint_callback.best_model_path

        return self.trainer.callback_metrics

    def test(
        self,
        checkpoint: str | Path | None = None,
        datamodule: EVAL_DATALOADERS | OTXDataModule | None = None,
        **kwargs,
    ) -> dict:
        """Run the testing phase of the engine.

        Args:
            datamodule (EVAL_DATALOADERS | OTXDataModule | None, optional): The data module containing the test data.
            checkpoint (str | Path | None, optional): Path to the checkpoint file to load the model from.
                Defaults to None.
            **kwargs: Additional keyword arguments for pl.Trainer configuration.

        Returns:
            dict: Dictionary containing the callback metrics from the trainer.

        Example:
            >>> engine.test(
            ...     datamodule=OTXDataModule(),
            ...     checkpoint=<checkpoint/path>,
            ... )

        CLI Usage:
            1. you can pick a model.
                ```python
                otx test
                    --model <CONFIG | CLASS_PATH_OR_NAME> --data_root <DATASET_PATH, str>
                    --checkpoint <CKPT_PATH, str>
                ```
            2. If you have a ready configuration file, run it like this.
                ```python
                otx test --config <CONFIG_PATH, str> --checkpoint <CKPT_PATH, str>
                ```
        """
        lit_module = self._build_lightning_module(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
        )
        if datamodule is None:
            datamodule = self.datamodule
        lit_module.meta_info = datamodule.meta_info

        self._build_trainer(**kwargs)

        self.trainer.test(
            model=lit_module,
            dataloaders=datamodule,
            ckpt_path=str(checkpoint) if checkpoint is not None else self.checkpoint,
        )

        return self.trainer.callback_metrics

    def predict(
        self,
        checkpoint: str | Path | None = None,
        datamodule: EVAL_DATALOADERS | OTXDataModule | None = None,
        return_predictions: bool | None = None,
        **kwargs,
    ) -> list | None:
        """Run predictions using the specified model and data.

        Args:
            datamodule (EVAL_DATALOADERS | OTXDataModule | None, optional): The data module to use for predictions.
            checkpoint (str | Path | None, optional): The path to the checkpoint file to load the model from.
            return_predictions (bool | None, optional): Whether to return the predictions or not.
            **kwargs: Additional keyword arguments for pl.Trainer configuration.

        Returns:
            list | None: The predictions if `return_predictions` is True, otherwise None.

        Example:
            >>> engine.predict(
            ...     datamodule=OTXDataModule(),
            ...     checkpoint=<checkpoint/path>,
            ...     return_predictions=True,
            ... )

        CLI Usage:
            1. you can pick a model.
                ```python
                otx predict
                    --model <CONFIG | CLASS_PATH_OR_NAME> --data_root <DATASET_PATH, str>
                    --checkpoint <CKPT_PATH, str>
                ```
            2. If you have a ready configuration file, run it like this.
                ```python
                otx predict --config <CONFIG_PATH, str> --checkpoint <CKPT_PATH, str>
                ```
        """
        lit_module = self._build_lightning_module(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
        )

        self._build_trainer(**kwargs)

        return self.trainer.predict(
            model=lit_module,
            datamodule=datamodule if datamodule is not None else self.datamodule,
            ckpt_path=str(checkpoint) if checkpoint is not None else self.checkpoint,
            return_predictions=return_predictions,
        )

    def export(self, *args, **kwargs) -> None:
        """Export the trained model to OpenVINO Intermediate Representation (IR) or ONNX formats."""
        raise NotImplementedError

    def explain(
        self,
        checkpoint: str | Path | None = None,
        datamodule: EVAL_DATALOADERS | OTXDataModule | None = None,
        explain_config: ExplainConfig | None = None,
        **kwargs,
    ) -> list | None:
        """Run XAI using the specified model and data.

        Args:
            checkpoint (str | Path | None, optional): The path to the checkpoint file to load the model from.
            datamodule (EVAL_DATALOADERS | OTXDataModule | None, optional): The data module to use for predictions.
            explain_config (ExplainConfig | None, optional): Config used to handle saliency maps.
            **kwargs: Additional keyword arguments for pl.Trainer configuration.

        Returns:
            list: Saliency maps.

        Example:
            >>> engine.explain(
            ...     datamodule=OTXDataModule(),
            ...     checkpoint=<checkpoint/path>,
            ...     explain_config=ExplainConfig(),
            ... )
        """
        from pathlib import Path

        import cv2

        ckpt_path = str(checkpoint) if checkpoint is not None else self.checkpoint
        if explain_config is None:
            explain_config = ExplainConfig()

        lit_module = self._build_lightning_module(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
        )
        if datamodule is None:
            datamodule = self.datamodule
        lit_module.meta_info = datamodule.meta_info

        lit_module.model.register_explain_hook()

        self._build_trainer(**kwargs)

        self.trainer.predict(
            model=lit_module,
            datamodule=datamodule,
            ckpt_path=ckpt_path,
        )
        saliency_maps = self.trainer.model.model.explain_hook.records
        cv2.imwrite(str(Path(self.work_dir) / "saliency_map.tiff"), saliency_maps[0][0])
        return saliency_maps

    # ------------------------------------------------------------------------ #
    # Property and setter functions provided by Engine.
    # ------------------------------------------------------------------------ #

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
            self.work_dir = self._trainer.default_root_dir

    @property
    def trainer_params(self) -> dict:
        """Returns the parameters used for training the model.

        Returns:
            dict: A dictionary containing the training parameters.
        """
        return self._cache.args

    @property
    def model(self) -> OTXModel:
        """Returns the model object associated with the engine.

        Returns:
            OTXModel: The OTXModel object.
        """
        return self._model

    @model.setter
    def model(self, model: OTXModel | str) -> None:
        """Sets the model for the engine.

        Args:
            model (OTXModel): The model to be set.

        Returns:
            None
        """
        if isinstance(model, str):
            # [TODO] harimkang: It will be updated in next PR.
            msg = "Auto-Configuration is not implemented yet."
            raise NotImplementedError(msg)
            # model = self._auto_configurator.get_model(model)
        self._model = model

    def _build_lightning_module(
        self,
        model: OTXModel,
        optimizer: OptimizerCallable,
        scheduler: LRSchedulerCallable,
    ) -> OTXLitModule:
        """Builds a LightningModule for engine workflow.

        Args:
            model (OTXModel): The OTXModel instance.
            optimizer (OptimizerCallable): The optimizer callable.
            scheduler (LRSchedulerCallable): The learning rate scheduler callable.

        Returns:
            OTXLitModule: The built LightningModule instance.
        """
        class_module, class_name = LITMODULE_PER_TASK[self.task].rsplit(".", 1)
        module = __import__(class_module, fromlist=[class_name])
        lightning_module = getattr(module, class_name)
        return lightning_module(
            otx_model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            torch_compile=False,
        )
