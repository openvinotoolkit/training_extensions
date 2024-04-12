# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module for OTX engine components."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Literal
from warnings import warn

import torch
from lightning import Trainer, seed_everything

from otx.algo.plugins import MixedPrecisionXPUPlugin
from otx.core.config.device import DeviceConfig
from otx.core.config.explain import ExplainConfig
from otx.core.config.hpo import HpoConfig
from otx.core.data.module import OTXDataModule
from otx.core.model.entity.base import OTXModel, OVModel
from otx.core.model.module.base import OTXLitModule
from otx.core.types import PathLike
from otx.core.types.device import DeviceType
from otx.core.types.export import OTXExportFormatType
from otx.core.types.precision import OTXPrecisionType
from otx.core.types.task import OTXTaskType
from otx.core.utils.cache import TrainerArgumentsCache
from otx.utils.utils import is_xpu_available

from .hpo import execute_hpo, update_hyper_parameter
from .utils.auto_configurator import DEFAULT_CONFIG_PER_TASK, AutoConfigurator

if TYPE_CHECKING:
    from lightning import Callback
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
    from lightning.pytorch.loggers import Logger
    from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
    from pytorch_lightning.trainer.connectors.accelerator_connector import _PRECISION_INPUT
    from torchmetrics import Metric

    from otx.core.metrics import MetricCallable


LITMODULE_PER_TASK = {
    OTXTaskType.MULTI_CLASS_CLS: "otx.core.model.module.classification.OTXMulticlassClsLitModule",
    OTXTaskType.MULTI_LABEL_CLS: "otx.core.model.module.classification.OTXMultilabelClsLitModule",
    OTXTaskType.H_LABEL_CLS: "otx.core.model.module.classification.OTXHlabelClsLitModule",
    OTXTaskType.DETECTION: "otx.core.model.module.detection.OTXDetectionLitModule",
    OTXTaskType.ROTATED_DETECTION: "otx.core.model.module.rotated_detection.OTXRotatedDetLitModule",
    OTXTaskType.INSTANCE_SEGMENTATION: "otx.core.model.module.instance_segmentation.OTXInstanceSegLitModule",
    OTXTaskType.SEMANTIC_SEGMENTATION: "otx.core.model.module.segmentation.OTXSegmentationLitModule",
    OTXTaskType.ACTION_CLASSIFICATION: "otx.core.model.module.action_classification.OTXActionClsLitModule",
    OTXTaskType.ACTION_DETECTION: "otx.core.model.module.action_detection.OTXActionDetLitModule",
    OTXTaskType.VISUAL_PROMPTING: "otx.core.model.module.visual_prompting.OTXVisualPromptingLitModule",
    OTXTaskType.ZERO_SHOT_VISUAL_PROMPTING: "otx.core.model.module.visual_prompting.OTXZeroShotVisualPromptingLitModule",  # noqa: E501
}


class Engine:
    """OTX Engine.

    This class defines the Engine for OTX, which governs each step of the OTX workflow.

    Example:
        The following examples show how to use the Engine class.

        Auto-Configuration with data_root::

            engine = Engine(
                data_root=<dataset/path>,
            )

        Create Engine with Custom OTXModel::

            engine = Engine(
                data_root=<dataset/path>,
                model=OTXModel(...),
                checkpoint=<checkpoint/path>,
            )

        Create Engine with Custom OTXDataModule::

            engine = Engine(
                model = OTXModel(...),
                datamodule = OTXDataModule(...),
            )
    """

    def __init__(
        self,
        *,
        data_root: PathLike | None = None,
        task: OTXTaskType | None = None,
        work_dir: PathLike = "./otx-workspace",
        datamodule: OTXDataModule | None = None,
        model: OTXModel | str | None = None,
        optimizer: list[OptimizerCallable] | OptimizerCallable | None = None,
        scheduler: list[LRSchedulerCallable] | LRSchedulerCallable | None = None,
        checkpoint: PathLike | None = None,
        device: DeviceType = DeviceType.auto,
        **kwargs,
    ):
        """Initializes the OTX Engine.

        Args:
            data_root (PathLike | None, optional): Root directory for the data. Defaults to None.
            task (OTXTaskType | None, optional): The type of OTX task. Defaults to None.
            work_dir (PathLike, optional): Working directory for the engine. Defaults to "./otx-workspace".
            datamodule (OTXDataModule | None, optional): The data module for the engine. Defaults to None.
            model (OTXModel | str | None, optional): The model for the engine. Defaults to None.
            optimizer (list[OptimizerCallable] | OptimizerCallable | None, optional): The optimizer for the engine.
                Defaults to None.
            scheduler (list[LRSchedulerCallable] | LRSchedulerCallable | None, optional):
                The learning rate scheduler for the engine. Defaults to None.
            checkpoint (PathLike | None, optional): Path to the checkpoint file. Defaults to None.
            device (DeviceType, optional): The device type to use. Defaults to DeviceType.auto.
            **kwargs: Additional keyword arguments for pl.Trainer.
        """
        self._cache = TrainerArgumentsCache(**kwargs)
        self.checkpoint = checkpoint
        self.work_dir = work_dir
        self.device = device  # type: ignore[assignment]
        self._auto_configurator = AutoConfigurator(
            data_root=data_root,
            task=datamodule.task if datamodule is not None else task,
            model_name=None if isinstance(model, OTXModel) else model,
        )

        self._datamodule: OTXDataModule | None = (
            datamodule if datamodule is not None else self._auto_configurator.get_datamodule()
        )
        self.task = task if task is not None else self._auto_configurator.task

        self._trainer: Trainer | None = None
        self._model: OTXModel = (
            model
            if isinstance(model, OTXModel)
            else self._auto_configurator.get_model(
                label_info=self._datamodule.label_info if self._datamodule is not None else None,
            )
        )

        self.optimizer: list[OptimizerCallable] | OptimizerCallable | None = (
            optimizer if optimizer is not None else self._auto_configurator.get_optimizer()
        )
        self.scheduler: list[LRSchedulerCallable] | LRSchedulerCallable | None = (
            scheduler if scheduler is not None else self._auto_configurator.get_scheduler()
        )

    _EXPORTED_MODEL_BASE_NAME = "exported_model"

    # ------------------------------------------------------------------------ #
    # General OTX Entry Points
    # ------------------------------------------------------------------------ #

    def train(
        self,
        max_epochs: int = 10,
        seed: int | None = None,
        deterministic: bool | Literal["warn"] = False,
        precision: _PRECISION_INPUT | None = "32",
        val_check_interval: int | float | None = None,
        callbacks: list[Callback] | Callback | None = None,
        logger: Logger | Iterable[Logger] | bool | None = None,
        resume: bool = False,
        metric: Metric | MetricCallable | None = None,
        run_hpo: bool = False,
        hpo_config: HpoConfig = HpoConfig(),  # noqa: B008 https://github.com/omni-us/jsonargparse/issues/423
        **kwargs,
    ) -> dict[str, Any]:
        """Trains the model using the provided LightningModule and OTXDataModule.

        Args:
            max_epochs (int | None, optional): The maximum number of epochs. Defaults to None.
            seed (int | None, optional): The random seed. Defaults to None.
            deterministic (bool | Literal["warn"]): Whether to enable deterministic behavior.
            Also, can be set to `warn` to avoid failures, because some operations don't
            support deterministic mode. Defaults to False.
            precision (_PRECISION_INPUT | None, optional): The precision of the model. Defaults to 32.
            val_check_interval (int | float | None, optional): The validation check interval. Defaults to None.
            callbacks (list[Callback] | Callback | None, optional): The callbacks to be used during training.
            logger (Logger | Iterable[Logger] | bool | None, optional): The logger(s) to be used. Defaults to None.
            resume (bool, optional): If True, tries to resume training from existing checkpoint.
            metric (Metric | MetricCallable | None): The metric for the validation and test.
                                            It could be None at export, predict, etc.
            run_hpo (bool, optional): If True, optimizer hyper parameters before training a model.
            hpo_config (HpoConfig | None, optional): Configuration for HPO.
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
        metric = metric if metric is not None else self._auto_configurator.get_metric()
        if run_hpo:
            best_config, best_trial_weight = execute_hpo(engine=self, **locals())
            if best_config is not None:
                update_hyper_parameter(self, best_config)
            if best_trial_weight is not None:
                self.checkpoint = best_trial_weight
                resume = True

        lit_module = self._build_lightning_module(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            metric=metric,
        )
        lit_module.label_info = self.datamodule.label_info

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
        checkpoint: PathLike | None = None,
        datamodule: EVAL_DATALOADERS | OTXDataModule | None = None,
        metric: Metric | MetricCallable | None = None,
        **kwargs,
    ) -> dict:
        """Run the testing phase of the engine.

        Args:
            datamodule (EVAL_DATALOADERS | OTXDataModule | None, optional): The data module containing the test data.
            checkpoint (PathLike | None, optional): Path to the checkpoint file to load the model from.
                Defaults to None.
            metric (Metric | MetricCallable | None): The metric for the validation and test.
                                            It could be None at export, predict, etc.
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
        model = self.model
        checkpoint = checkpoint if checkpoint is not None else self.checkpoint
        datamodule = datamodule if datamodule is not None else self.datamodule

        is_ir_ckpt = Path(str(checkpoint)).suffix in [".xml", ".onnx"]
        if is_ir_ckpt and not isinstance(model, OVModel):
            model = self._auto_configurator.get_ov_model(model_name=str(checkpoint), label_info=datamodule.label_info)
            if self.device.accelerator != "cpu":
                msg = "IR model supports inference only on CPU device. The device is changed automatic."
                warn(msg, stacklevel=1)
                self.device = DeviceType.cpu  # type: ignore[assignment]

        # NOTE: Re-initiate datamodule for OVModel as model API supports its own data pipeline.
        if isinstance(model, OVModel):
            datamodule = self._auto_configurator.update_ov_subset_pipeline(datamodule=datamodule, subset="test")

        metric = metric if metric is not None else self._auto_configurator.get_metric()
        lit_module = self._build_lightning_module(
            model=model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            metric=metric,
        )
        lit_module.label_info = datamodule.label_info

        # NOTE, trainer.test takes only lightning based checkpoint.
        # So, it can't take the OTX1.x checkpoint.
        if checkpoint is not None and not is_ir_ckpt:
            loaded_checkpoint = torch.load(checkpoint)
            lit_module.load_state_dict(loaded_checkpoint)

        self._build_trainer(**kwargs)

        self.trainer.test(
            model=lit_module,
            dataloaders=datamodule,
        )

        return self.trainer.callback_metrics

    def predict(
        self,
        checkpoint: PathLike | None = None,
        datamodule: EVAL_DATALOADERS | OTXDataModule | None = None,
        return_predictions: bool | None = None,
        explain: bool = False,
        explain_config: ExplainConfig | None = None,
        **kwargs,
    ) -> list | None:
        """Run predictions using the specified model and data.

        Args:
            datamodule (EVAL_DATALOADERS | OTXDataModule | None, optional): The data module to use for predictions.
            checkpoint (PathLike | None, optional): The path to the checkpoint file to load the model from.
            return_predictions (bool | None, optional): Whether to return the predictions or not.
            explain (bool): Whether to dump "saliency_map" and "feature_vector" or not.
            explain_config (ExplainConfig): Explain configuration (used for saliency map post-processing).
            **kwargs: Additional keyword arguments for pl.Trainer configuration.

        Returns:
            list | None: The predictions if `return_predictions` is True, otherwise None.

        Example:
            >>> engine.predict(
            ...     datamodule=OTXDataModule(),
            ...     checkpoint=<checkpoint/path>,
            ...     return_predictions=True,
            ...     explain=True,
            ... )

        CLI Usage:
            1. you can pick a model.
                ```python
                otx predict
                    --config <CONFIG_PATH> --data_root <DATASET_PATH, str>
                    --checkpoint <CKPT_PATH, str>
                ```
            2. If you have a ready configuration file, run it like this.
                ```python
                otx predict --config <CONFIG_PATH, str> --checkpoint <CKPT_PATH, str>
                ```
        """
        from otx.algo.utils.xai_utils import process_saliency_maps_in_pred_entity

        model = self.model

        checkpoint = checkpoint if checkpoint is not None else self.checkpoint
        datamodule = datamodule if datamodule is not None else self.datamodule

        is_ir_ckpt = checkpoint is not None and Path(checkpoint).suffix in [".xml", ".onnx"]
        if is_ir_ckpt and not isinstance(model, OVModel):
            model = self._auto_configurator.get_ov_model(model_name=str(checkpoint), label_info=datamodule.label_info)

        # NOTE: Re-initiate datamodule for OVModel as model API supports its own data pipeline.
        if isinstance(model, OVModel):
            datamodule = self._auto_configurator.update_ov_subset_pipeline(datamodule=datamodule, subset="test")

        lit_module = self._build_lightning_module(
            model=model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
        )
        lit_module.label_info = datamodule.label_info

        if checkpoint is not None and not is_ir_ckpt:
            loaded_checkpoint = torch.load(checkpoint)
            lit_module.load_state_dict(loaded_checkpoint)

        lit_module.model.explain_mode = explain

        self._build_trainer(**kwargs)

        predict_result = self.trainer.predict(
            model=lit_module,
            dataloaders=datamodule,
            return_predictions=return_predictions,
        )

        if explain:
            if explain_config is None:
                explain_config = ExplainConfig()

            predict_result = process_saliency_maps_in_pred_entity(predict_result, explain_config)

        lit_module.model.explain_mode = False
        return predict_result

    def export(
        self,
        checkpoint: str | Path | None = None,
        export_format: OTXExportFormatType = OTXExportFormatType.OPENVINO,
        export_precision: OTXPrecisionType = OTXPrecisionType.FP32,
        explain: bool = False,
    ) -> Path:
        """Export the trained model to OpenVINO Intermediate Representation (IR) or ONNX formats.

        Args:
            checkpoint (str | Path | None, optional): Checkpoint to export. Defaults to None.
            export_config (ExportConfig | None, optional): Config that allows to set export
            format and precision. Defaults to None.
            explain (bool): Whether to get "saliency_map" and "feature_vector" or not.

        Returns:
            Path: Path to the exported model.

        Example:
            >>> engine.export(
            ...     checkpoint=<checkpoint/path>,
            ...     export_format=OTXExportFormatType.OPENVINO,
            ...     export_precision=OTXExportPrecisionType.FP32,
            ...     explain=True,
            ... )

        CLI Usage:
            1. To export a model with default setting (OPENVINO, FP32), run
                ```python
                otx export
                    --config <CONFIG_PATH> --data_root <DATASET_PATH, str>
                    --checkpoint <CKPT_PATH, str>
                ```
            2. To export a model with precision FP16 and format ONNX, run
                ```python
                otx export
                    --config <CONFIG_PATH> --data_root <DATASET_PATH, str>
                    --checkpoint <CKPT_PATH, str> --export_precision FP16 --export_format ONNX
                ```
        """
        ckpt_path = str(checkpoint) if checkpoint is not None else self.checkpoint

        if ckpt_path is None:
            msg = "To make export, checkpoint must be specified."
            raise RuntimeError(msg)

        self.model.eval()
        lit_module = self._build_lightning_module(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
        )
        loaded_checkpoint = torch.load(ckpt_path)
        lit_module.label_info = loaded_checkpoint["state_dict"]["label_info"]
        self.model.label_info = lit_module.label_info

        lit_module.load_state_dict(loaded_checkpoint)

        self.model.explain_mode = explain

        exported_model_path = lit_module.export(
            output_dir=Path(self.work_dir),
            base_name=self._EXPORTED_MODEL_BASE_NAME,
            export_format=export_format,
            precision=export_precision,
        )

        self.model.explain_mode = False
        return exported_model_path

    def optimize(
        self,
        checkpoint: PathLike | None = None,
        datamodule: TRAIN_DATALOADERS | OTXDataModule | None = None,
        max_data_subset_size: int | None = None,
    ) -> Path:
        """Applies NNCF.PTQ to the underlying models (now works only for OV models).

        PTQ performs int-8 quantization on the input model, so the resulting model
        comes in mixed precision (some operations, however, remain in FP32).

        Args:
            checkpoint (str | Path | None, optional): Checkpoint to optimize. Defaults to None.
            datamodule (TRAIN_DATALOADERS | OTXDataModule | None, optional): The data module to use for optimization.
            max_data_subset_size (int | None): The maximum size of the train subset from `datamodule` that would be
            used for model optimization. If not set, NNCF.PTQ will select subset size according to it's
            default settings.

        Returns:
            Path: path to the optimized model.

        Example:
            >>> engine.optimize(
            ...     checkpoint=<checkpoint/path>,
            ...     datamodule=OTXDataModule(),
            ...     checkpoint=<checkpoint/path>,
            ... )

        CLI Usage:
            To optimize a model, run
                ```python
                otx optimize
                    --checkpoint <CKPT_PATH, str>
                    --model <CONFIG | CLASS_PATH_OR_NAME> --data_root <DATASET_PATH, str>
                    --model.model_name=<PATH_TO_IR_XML, str>
                ```
        """
        checkpoint = checkpoint if checkpoint is not None else self.checkpoint
        optimize_datamodule = datamodule if datamodule is not None else self.datamodule

        is_ir_ckpt = checkpoint is not None and Path(checkpoint).suffix in [".xml", ".onnx"]
        if not is_ir_ckpt:
            msg = "Engine.optimize() supports only OV IR or ONNX checkpoints"
            raise RuntimeError(msg)

        model = self.model
        if not isinstance(model, OVModel):
            optimize_datamodule = self._auto_configurator.update_ov_subset_pipeline(
                datamodule=optimize_datamodule,
                subset="train",
            )
            model = self._auto_configurator.get_ov_model(
                model_name=str(checkpoint),
                label_info=optimize_datamodule.label_info,
            )

        ptq_config = {}
        if max_data_subset_size is not None:
            ptq_config["subset_size"] = max_data_subset_size

        return model.optimize(
            Path(self.work_dir),
            optimize_datamodule,
            ptq_config,
        )

    def explain(
        self,
        checkpoint: PathLike | None = None,
        datamodule: EVAL_DATALOADERS | OTXDataModule | None = None,
        explain_config: ExplainConfig | None = None,
        dump: bool | None = False,
        **kwargs,
    ) -> list | None:
        """Run XAI using the specified model and data (test subset).

        Args:
            checkpoint (PathLike | None, optional): The path to the checkpoint file to load the model from.
            datamodule (EVAL_DATALOADERS | OTXDataModule | None, optional): The data module to use for predictions.
            explain_config (ExplainConfig | None, optional): Config used to handle saliency maps.
            dump (bool): Whether to dump "saliency_map" or not.
            **kwargs: Additional keyword arguments for pl.Trainer configuration.

        Returns:
            list: Saliency maps.

        Example:
            >>> engine.explain(
            ...     datamodule=OTXDataModule(),
            ...     checkpoint=<checkpoint/path>,
            ...     explain_config=ExplainConfig(),
            ...     dump=True,
            ... )

        CLI Usage:
            1. To run XAI using the specified model, run
                ```python
                otx explain
                    --config <CONFIG_PATH> --data_root <DATASET_PATH, str>
                    --checkpoint <CKPT_PATH, str>
                ```
        """
        from otx.algo.utils.xai_utils import dump_saliency_maps, process_saliency_maps_in_pred_entity

        model = self.model

        checkpoint = checkpoint if checkpoint is not None else self.checkpoint
        datamodule = datamodule if datamodule is not None else self.datamodule

        is_ir_ckpt = checkpoint is not None and Path(checkpoint).suffix in [".xml", ".onnx"]
        if is_ir_ckpt and not isinstance(model, OVModel):
            datamodule = self._auto_configurator.update_ov_subset_pipeline(datamodule=datamodule, subset="test")
            model = self._auto_configurator.get_ov_model(model_name=str(checkpoint), label_info=datamodule.label_info)

        lit_module = self._build_lightning_module(
            model=model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
        )
        lit_module.label_info = datamodule.label_info

        if checkpoint is not None and not is_ir_ckpt:
            loaded_checkpoint = torch.load(checkpoint)
            lit_module.load_state_dict(loaded_checkpoint)

        lit_module.model.explain_mode = True

        self._build_trainer(**kwargs)

        predict_result = self.trainer.predict(
            model=lit_module,
            datamodule=datamodule,
        )

        if explain_config is None:
            explain_config = ExplainConfig()

        predict_result = process_saliency_maps_in_pred_entity(predict_result, explain_config)
        if dump:
            dump_saliency_maps(
                predict_result,
                explain_config,
                datamodule,
                output_dir=Path(self.work_dir),
            )
        lit_module.model.explain_mode = False
        return predict_result

    @classmethod
    def from_config(
        cls,
        config_path: PathLike,
        data_root: PathLike | None = None,
        work_dir: PathLike | None = None,
        **kwargs,
    ) -> Engine:
        """Builds the engine from a configuration file.

        Args:
            config_path (PathLike): The configuration file path.
            data_root (PathLike | None): Root directory for the data.
                Defaults to None. If data_root is None, use the data_root from the configuration file.
            work_dir (PathLike | None, optional): Working directory for the engine.
                Defaults to None. If work_dir is None, use the work_dir from the configuration file.
            kwargs: Arguments that can override the engine's arguments.

        Returns:s
            Engine: An instance of the Engine class.

        Example:
            >>> engine = Engine.from_config(
            ...     config="config.yaml",
            ... )
        """
        from otx.cli.utils.jsonargparse import get_instantiated_classes

        # For the Engine argument, prepend 'engine.' for CLI parser
        filter_kwargs = ["device", "checkpoint", "task"]
        for key in filter_kwargs:
            if key in kwargs:
                kwargs[f"engine.{key}"] = kwargs.pop(key)
        instantiated_config, train_kwargs = get_instantiated_classes(
            config=config_path,
            data_root=data_root,
            work_dir=work_dir,
            **kwargs,
        )
        engine_kwargs = {**instantiated_config.get("engine", {}), **train_kwargs}

        # Remove any input that is not currently available in Engine and print a warning message.
        set_valid_args = TrainerArgumentsCache.get_trainer_constructor_args().union(
            set(inspect.signature(Engine.__init__).parameters.keys()),
        )
        removed_args = []
        for engine_key in list(engine_kwargs.keys()):
            if engine_key not in set_valid_args:
                engine_kwargs.pop(engine_key)
                removed_args.append(engine_key)
        if removed_args:
            msg = (
                f"Warning: {removed_args} -> not available in Engine constructor. "
                "It will be ignored. Use what need in the right places."
            )
            warn(msg, stacklevel=1)

        return cls(
            work_dir=instantiated_config.get("work_dir", work_dir),
            datamodule=instantiated_config.get("data"),
            model=instantiated_config.get("model"),
            optimizer=instantiated_config.get("optimizer"),
            scheduler=instantiated_config.get("scheduler"),
            **engine_kwargs,
        )

    @classmethod
    def from_model_name(
        cls,
        model_name: str,
        task: OTXTaskType,
        data_root: PathLike | None = None,
        work_dir: PathLike | None = None,
        **kwargs,
    ) -> Engine:
        """Builds the engine from a model name.

        Args:
            model_name (str): The model name.
            task (OTXTaskType): The type of OTX task.
            data_root (PathLike | None): Root directory for the data.
                Defaults to None. If data_root is None, use the data_root from the configuration file.
            work_dir (PathLike | None, optional): Working directory for the engine.
                Defaults to None. If work_dir is None, use the work_dir from the configuration file.
            kwargs: Arguments that can override the engine's arguments.

        Returns:
            Engine: An instance of the Engine class.

        Example:
            >>> engine = Engine.from_model_name(
            ...     model_name="atss_mobilenetv2",
            ...     task="DETECTION",
            ...     data_root=<dataset/path>,
            ... )

            If you want to override configuration from default config:
                >>> overriding = {
                ...     "data.config.train_subset.batch_size": 2,
                ...     "data.config.test_subset.subset_name": "TESTING",
                ... }
                >>> engine = Engine(
                ...     model_name="atss_mobilenetv2",
                ...     task="DETECTION",
                ...     data_root=<dataset/path>,
                ...     **overriding,
                ... )
        """
        default_config = DEFAULT_CONFIG_PER_TASK.get(task)
        model_path = str(default_config).split("/")
        model_path[-1] = f"{model_name}.yaml"
        config = Path("/".join(model_path))
        if not config.exists():
            candidate_list = [model.stem for model in config.parent.glob("*")]
            msg = (
                f"Model config file not found: {config}, please check the model name. "
                f"Available models for {task} task are {candidate_list}"
            )
            raise FileNotFoundError(msg)

        return cls.from_config(
            config_path=config,
            data_root=data_root,
            work_dir=work_dir,
            task=task,
            **kwargs,
        )

    # ------------------------------------------------------------------------ #
    # Property and setter functions provided by Engine.
    # ------------------------------------------------------------------------ #

    @property
    def work_dir(self) -> PathLike:
        """Work directory."""
        return self._work_dir

    @work_dir.setter
    def work_dir(self, work_dir: PathLike) -> None:
        self._work_dir = work_dir
        self._cache.update(default_root_dir=work_dir)
        self._cache.is_trainer_args_identical = False

    @property
    def device(self) -> DeviceConfig:
        """Device engine uses."""
        return self._device

    @device.setter
    def device(self, device: DeviceType) -> None:
        if is_xpu_available() and device == DeviceType.auto:
            device = DeviceType.xpu
        self._device = DeviceConfig(accelerator=device)
        self._cache.update(accelerator=self._device.accelerator, devices=self._device.devices)
        self._cache.is_trainer_args_identical = False

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
            # set up xpu device
            if self._device.accelerator == DeviceType.xpu:
                self._cache.update(strategy="xpu_single")
                # add plugin for Automatic Mixed Precision on XPU
                if self._cache.args.get("precision", 32) == 16:
                    self._cache.update(plugins=[MixedPrecisionXPUPlugin()])
                    self._cache.args["precision"] = None

            kwargs = self._cache.args
            self._trainer = Trainer(**kwargs)
            self._cache.is_trainer_args_identical = True
            self._trainer.task = self.task
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
            model (OTXModel | str): The model to be set.

        Returns:
            None
        """
        if isinstance(model, str):
            model = self._auto_configurator.get_model(model, label_info=self.datamodule.label_info)
        self._model = model

    @property
    def datamodule(self) -> OTXDataModule:
        """Returns the datamodule object associated with the engine.

        Returns:
            OTXDataModule: The OTXDataModule object.
        """
        if self._datamodule is None:
            msg = "Please include the `data_root` or `datamodule` when creating the Engine."
            raise RuntimeError(msg)
        return self._datamodule

    def _build_lightning_module(
        self,
        model: OTXModel,
        optimizer: list[OptimizerCallable] | OptimizerCallable | None,
        scheduler: list[LRSchedulerCallable] | LRSchedulerCallable | None,
        metric: Metric | MetricCallable | None = None,
    ) -> OTXLitModule:
        """Builds a LightningModule for engine workflow.

        Args:
            model (OTXModel): The OTXModel instance.
            optimizer (list[OptimizerCallable] | OptimizerCallable | None): The optimizer callable.
            scheduler (list[LRSchedulerCallable] | LRSchedulerCallable | None): The learning rate scheduler callable.
            metric (Metric | MetricCallable | None): The metric for the validation and test.
                                            It could be None at export, predict, etc.

        Returns:
            OTXLitModule | OTXModel: The built LightningModule instance.
        """
        if self.task in (
            OTXTaskType.ANOMALY_CLASSIFICATION,
            OTXTaskType.ANOMALY_DETECTION,
            OTXTaskType.ANOMALY_SEGMENTATION,
        ):
            model = self._get_anomaly_model(model, optimizer, scheduler)
        else:
            class_module, class_name = LITMODULE_PER_TASK[self.task].rsplit(".", 1)
            module = __import__(class_module, fromlist=[class_name])
            lightning_module = getattr(module, class_name)
            lightning_kwargs = {
                "otx_model": model,
                "optimizer": optimizer,
                "scheduler": scheduler,
                "torch_compile": False,
            }
            if metric:
                lightning_kwargs["metric"] = metric

            model = lightning_module(**lightning_kwargs)
        return model

    def _get_anomaly_model(
        self,
        model: OTXModel,
        optimizer: list[OptimizerCallable] | OptimizerCallable | None,
        scheduler: list[LRSchedulerCallable] | LRSchedulerCallable | None,
    ) -> OTXModel:
        # [TODO](ashwinvaidya17): Need to revisit how task, optimizer, and scheduler are assigned to the model
        model.task = self.task
        model.optimizer = optimizer
        model.scheduler = scheduler
        return model
