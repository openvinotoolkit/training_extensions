# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module for OTX engine components."""

from __future__ import annotations

import copy
import csv
import inspect
import logging
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Iterable, Iterator, Literal
from warnings import warn

import torch
from lightning import Trainer, seed_everything

from otx.algo.plugins import MixedPrecisionXPUPlugin
from otx.core.config.device import DeviceConfig
from otx.core.config.explain import ExplainConfig
from otx.core.config.hpo import HpoConfig
from otx.core.data.module import OTXDataModule
from otx.core.model.base import OTXModel, OVModel
from otx.core.types import PathLike
from otx.core.types.device import DeviceType
from otx.core.types.export import OTXExportFormatType
from otx.core.types.precision import OTXPrecisionType
from otx.core.types.task import OTXTaskType
from otx.core.utils.cache import TrainerArgumentsCache
from otx.utils.utils import is_xpu_available, measure_flops

from .adaptive_bs import adapt_batch_size
from .hpo import execute_hpo, update_hyper_parameter
from .utils.auto_configurator import DEFAULT_CONFIG_PER_TASK, AutoConfigurator

if TYPE_CHECKING:
    from lightning import Callback
    from lightning.pytorch.loggers import Logger
    from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
    from pytorch_lightning.trainer.connectors.accelerator_connector import _PRECISION_INPUT

    from otx.core.metrics import MetricCallable


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

    _EXPORTED_MODEL_BASE_NAME: ClassVar[str] = "exported_model"

    def __init__(
        self,
        *,
        data_root: PathLike | None = None,
        task: OTXTaskType | None = None,
        work_dir: PathLike = "./otx-workspace",
        datamodule: OTXDataModule | None = None,
        model: OTXModel | str | None = None,
        checkpoint: PathLike | None = None,
        device: DeviceType = DeviceType.auto,
        num_devices: int = 1,
        **kwargs,
    ):
        """Initializes the OTX Engine.

        Args:
            data_root (PathLike | None, optional): Root directory for the data. Defaults to None.
            task (OTXTaskType | None, optional): The type of OTX task. Defaults to None.
            work_dir (PathLike, optional): Working directory for the engine. Defaults to "./otx-workspace".
            datamodule (OTXDataModule | None, optional): The data module for the engine. Defaults to None.
            model (OTXModel | str | None, optional): The model for the engine. Defaults to None.
            checkpoint (PathLike | None, optional): Path to the checkpoint file. Defaults to None.
            device (DeviceType, optional): The device type to use. Defaults to DeviceType.auto.
            num_devices (int, optional): The number of devices to use. If it is 2 or more, it will behave as multi-gpu.
            **kwargs: Additional keyword arguments for pl.Trainer.
        """
        self._cache = TrainerArgumentsCache(**kwargs)
        self.checkpoint = checkpoint
        self.work_dir = work_dir
        self.device = device  # type: ignore[assignment]
        self.num_devices = num_devices
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
        get_model_args: dict[str, Any] = {}
        if self._datamodule is not None:
            get_model_args["label_info"] = self._datamodule.label_info
            if (input_size := self._datamodule.input_size) is not None:
                get_model_args["input_size"] = (input_size, input_size) if isinstance(input_size, int) else input_size
        self._model: OTXModel = (
            model if isinstance(model, OTXModel) else self._auto_configurator.get_model(**get_model_args)
        )

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
        metric: MetricCallable | None = None,
        run_hpo: bool = False,
        hpo_config: HpoConfig = HpoConfig(),  # noqa: B008 https://github.com/omni-us/jsonargparse/issues/423
        checkpoint: PathLike | None = None,
        adaptive_bs: Literal["None", "Safe", "Full"] = "None",
        **kwargs,
    ) -> dict[str, Any]:
        r"""Trains the model using the provided LightningModule and OTXDataModule.

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
            metric (MetricCallable | None): If not None, it will override `OTXModel.metric_callable` with the given
                metric callable. It will temporarilly change the evaluation metric for the validation and test.
            run_hpo (bool, optional): If True, optimizer hyper parameters before training a model.
            hpo_config (HpoConfig | None, optional): Configuration for HPO.
            checkpoint (PathLike | None, optional): Path to the checkpoint file. Defaults to None.
            adaptive_bs (Literal["None", "Safe", "Full"]):
                Change the actual batch size depending on the current GPU status.
                Safe => Prevent GPU out of memory. Full => Find a batch size using most of GPU memory.
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
            1. Can train with data_root only. then OTX will provide default training configuration.
                ```shell
                >>> otx train --data_root <DATASET_PATH, str>
                ```
            2. Can pick a model or datamodule as Config file or Class.
                ```shell
                >>> otx train \
                ...     --data_root <DATASET_PATH, str> \
                ...     --model <CONFIG | CLASS_PATH_OR_NAME, OTXModel> \
                ...     --data <CONFIG | CLASS_PATH_OR_NAME, OTXDataModule>
                ```
            3. Of course, can override the various values with commands.
                ```shell
                >>> otx train \
                ...     --data_root <DATASET_PATH, str> \
                ...     --max_epochs <EPOCHS, int> \
                ...     --checkpoint <CKPT_PATH, str>
                ```
            4. To train with configuration file, run
                ```shell
                >>> otx train --data_root <DATASET_PATH, str> --config <CONFIG_PATH, str>
                ```
            5. To reproduce the existing training with work_dir, run
                ```shell
                >>> otx train --work_dir <WORK_DIR_PATH, str>
                ```
        """
        checkpoint = checkpoint if checkpoint is not None else self.checkpoint

        if adaptive_bs != "None":
            adapt_batch_size(engine=self, **locals(), not_increase=(adaptive_bs != "Full"))

        if run_hpo:
            best_config, best_trial_weight = execute_hpo(engine=self, **locals())
            if best_config is not None:
                update_hyper_parameter(self, best_config)
            if best_trial_weight is not None:
                checkpoint = best_trial_weight
                resume = True

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

        # NOTE: Model's label info should be converted datamodule's label info before ckpt loading
        # This is due to smart weight loading check label name as well as number of classes.
        if self.model.label_info != self.datamodule.label_info:
            msg = (
                "Model label_info is not equal to the Datamodule label_info. "
                f"It will be overriden: {self.model.label_info} => {self.datamodule.label_info}"
            )
            logging.warning(msg)
            self.model.label_info = self.datamodule.label_info

        if resume and checkpoint:
            # NOTE: If both `resume` and `checkpoint` are provided,
            # load the entire model state from the checkpoint using the pl.Trainer's API.
            fit_kwargs["ckpt_path"] = checkpoint
        elif not resume and checkpoint:
            # NOTE: If `resume` is not enabled but `checkpoint` is provided,
            # load the model state from the checkpoint incrementally.
            # This means only the model weights are loaded. If there is a mismatch in label_info,
            # perform incremental weight loading for the model's classification layer.
            ckpt = torch.load(checkpoint)
            self.model.load_state_dict_incrementally(ckpt)

        with override_metric_callable(model=self.model, new_metric_callable=metric) as model:
            self.trainer.fit(
                model=model,
                datamodule=self.datamodule,
                **fit_kwargs,
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

    def test(
        self,
        checkpoint: PathLike | None = None,
        datamodule: EVAL_DATALOADERS | OTXDataModule | None = None,
        metric: MetricCallable | None = None,
        **kwargs,
    ) -> dict:
        r"""Run the testing phase of the engine.

        Args:
            checkpoint (PathLike | None, optional): Path to the checkpoint file to load the model from.
                Defaults to None.
            datamodule (EVAL_DATALOADERS | OTXDataModule | None, optional): The data module containing the test data.
            metric (MetricCallable | None): If not None, it will override `OTXModel.metric_callable` with the given
                metric callable. It will temporarilly change the evaluation metric for the validation and test.
            **kwargs: Additional keyword arguments for pl.Trainer configuration.

        Returns:
            dict: Dictionary containing the callback metrics from the trainer.

        Example:
            >>> engine.test(
            ...     datamodule=OTXDataModule(),
            ...     checkpoint=<checkpoint/path>,
            ... )

        CLI Usage:
            1. To eval model by specifying the work_dir where did the training, run
                ```shell
                >>> otx test --work_dir <WORK_DIR_PATH, str>
                ```
            2. To eval model a specific checkpoint, run
                ```shell
                >>> otx test --work_dir <WORK_DIR_PATH, str> --checkpoint <CKPT_PATH, str>
                ```
            3. Can pick a model.
                ```shell
                >>> otx test \
                ...     --model <CONFIG | CLASS_PATH_OR_NAME> \
                ...     --data_root <DATASET_PATH, str> \
                ...     --checkpoint <CKPT_PATH, str>
                ```
            4. To eval with configuration file, run
                ```shell
                >>> otx test --config <CONFIG_PATH, str> --checkpoint <CKPT_PATH, str>
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

        # NOTE: Re-initiate datamodule without tiling as model API supports its own tiling mechanism
        if isinstance(model, OVModel):
            datamodule = self._auto_configurator.update_ov_subset_pipeline(datamodule=datamodule, subset="test")

        # NOTE, trainer.test takes only lightning based checkpoint.
        # So, it can't take the OTX1.x checkpoint.
        if checkpoint is not None and not is_ir_ckpt:
            kwargs_user_input: dict[str, Any] = {}
            if self.task == OTXTaskType.ZERO_SHOT_VISUAL_PROMPTING:
                # to update user's custom infer_reference_info_root through cli for zero-shot learning
                # TODO (sungchul): revisit for better solution
                kwargs_user_input.update(infer_reference_info_root=self.model.infer_reference_info_root)

            model_cls = model.__class__
            model = model_cls.load_from_checkpoint(checkpoint_path=checkpoint, **kwargs_user_input)

        if model.label_info != self.datamodule.label_info:
            if (
                self.task == "SEMANTIC_SEGMENTATION"
                and "otx_background_lbl" in self.datamodule.label_info.label_names
                and (len(self.datamodule.label_info.label_names) - len(model.label_info.label_names) == 1)
            ):
                # workaround for background label
                model.label_info = copy.deepcopy(self.datamodule.label_info)
            else:
                msg = (
                    "To launch a test pipeline, the label information should be same "
                    "between the training and testing datasets. "
                    "Please check whether you use the same dataset: "
                    f"model.label_info={model.label_info}, "
                    f"datamodule.label_info={self.datamodule.label_info}"
                )
                raise ValueError(msg)

        self._build_trainer(**kwargs)

        with override_metric_callable(model=model, new_metric_callable=metric) as model:
            self.trainer.test(
                model=model,
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
        r"""Run predictions using the specified model and data.

        Args:
            checkpoint (PathLike | None, optional): The path to the checkpoint file to load the model from.
            datamodule (EVAL_DATALOADERS | OTXDataModule | None, optional): The data module to use for predictions.
            return_predictions (bool | None, optional): Whether to return the predictions or not.
            explain (bool, optional): Whether to dump "saliency_map" and "feature_vector" or not.
            explain_config (ExplainConfig | None, optional): Explain configuration used for saliency map post-processing
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
            1. To predict a model with work_dir, run
                ```shell
                >>> otx predict --work_dir <WORK_DIR_PATH, str>
                ```
            2. To predict a specific model, run
                ```shell
                >>> otx predict \
                ...     --work_dir <WORK_DIR_PATH, str> \
                ...     --checkpoint <CKPT_PATH, str>
                ```
            3. To predict with configuration file, run
                ```shell
                >>> otx predict \
                ...     --config <CONFIG_PATH, str> \
                ...     --checkpoint <CKPT_PATH, str>
                ```
        """
        from otx.algo.utils.xai_utils import process_saliency_maps_in_pred_entity, set_crop_padded_map_flag

        model = self.model

        checkpoint = checkpoint if checkpoint is not None else self.checkpoint
        datamodule = datamodule if datamodule is not None else self.datamodule

        is_ir_ckpt = checkpoint is not None and Path(checkpoint).suffix in [".xml", ".onnx"]
        if is_ir_ckpt and not isinstance(model, OVModel):
            model = self._auto_configurator.get_ov_model(model_name=str(checkpoint), label_info=datamodule.label_info)

        # NOTE: Re-initiate datamodule for OVModel as model API supports its own data pipeline.
        if isinstance(model, OVModel):
            datamodule = self._auto_configurator.update_ov_subset_pipeline(datamodule=datamodule, subset="test")

        if checkpoint is not None and not is_ir_ckpt:
            kwargs_user_input: dict[str, Any] = {}
            if self.task == OTXTaskType.ZERO_SHOT_VISUAL_PROMPTING:
                # to update user's custom infer_reference_info_root through cli for zero-shot learning
                # TODO (sungchul): revisit for better solution
                kwargs_user_input.update(infer_reference_info_root=self.model.infer_reference_info_root)

            model_cls = model.__class__
            model = model_cls.load_from_checkpoint(checkpoint_path=checkpoint, **kwargs_user_input)

        if model.label_info != self.datamodule.label_info:
            msg = (
                "To launch a predict pipeline, the label information should be same "
                "between the training and testing datasets. "
                "Please check whether you use the same dataset: "
                f"model.label_info={model.label_info}, "
                f"datamodule.label_info={self.datamodule.label_info}"
            )
            raise ValueError(msg)

        self._build_trainer(**kwargs)

        curr_explain_mode = model.explain_mode

        try:
            model.explain_mode = explain
            predict_result = self.trainer.predict(
                model=model,
                dataloaders=datamodule,
                return_predictions=return_predictions,
            )
        finally:
            model.explain_mode = curr_explain_mode

        if explain:
            if explain_config is None:
                explain_config = ExplainConfig()
            explain_config = set_crop_padded_map_flag(explain_config, datamodule)

            predict_result = process_saliency_maps_in_pred_entity(predict_result, explain_config, datamodule.label_info)

        return predict_result

    def export(
        self,
        checkpoint: PathLike | None = None,
        export_format: OTXExportFormatType = OTXExportFormatType.OPENVINO,
        export_precision: OTXPrecisionType = OTXPrecisionType.FP32,
        explain: bool = False,
        export_demo_package: bool = False,
    ) -> Path:
        r"""Export the trained model to OpenVINO Intermediate Representation (IR) or ONNX formats.

        Args:
            checkpoint (PathLike | None, optional): Checkpoint to export. Defaults to None.
            export_config (ExportConfig | None, optional): Config that allows to set export
            format and precision. Defaults to None.
            explain (bool): Whether to get "saliency_map" and "feature_vector" or not.
            export_demo_package (bool): Whether to export demo package with the model.
                Only OpenVINO model can be exported with demo package.

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
                ```shell
                >>> otx export --work_dir <WORK_DIR_PATH, str>
                ```
            2. To export a specific checkpoint, run
                ```shell
                >>> otx export --config <CONFIG_PATH, str> --checkpoint <CKPT_PATH, str>
                ```
            3. To export a model with precision FP16 and format ONNX, run
                ```shell
                >>> otx export ... \
                ...     --export_precision FP16 --export_format ONNX
                ```
            4. To export model with 'saliency_map' and 'feature_vector', run
                ```shell
                >>> otx export ... \
                ...     --explain True
                ```
        """
        checkpoint = checkpoint if checkpoint is not None else self.checkpoint

        if checkpoint is None:
            msg = "To make export, checkpoint must be specified."
            raise RuntimeError(msg)
        is_ir_ckpt = Path(checkpoint).suffix in [".xml"]
        if export_demo_package and export_format == OTXExportFormatType.ONNX:
            msg = (
                "ONNX export is not supported in exportable code mode. "
                "Exportable code parameter will be disregarded. "
            )
            warn(msg, stacklevel=1)
            export_demo_package = False

        if is_ir_ckpt and not export_demo_package:
            msg = "IR model is passed as a checkpoint, export automatically switched to exportable code."
            warn(msg, stacklevel=1)
            export_demo_package = True

        if is_ir_ckpt and not isinstance(self.model, OVModel):
            # create OVModel
            self.model = self._auto_configurator.get_ov_model(
                model_name=str(checkpoint),
                label_info=self.datamodule.label_info,
            )

        if not is_ir_ckpt:
            kwargs_user_input: dict[str, Any] = {}
            if self.task == OTXTaskType.ZERO_SHOT_VISUAL_PROMPTING:
                # to update user's custom infer_reference_info_root through cli for zero-shot learning
                # TODO (sungchul): revisit for better solution
                kwargs_user_input.update(infer_reference_info_root=self.model.infer_reference_info_root)

            model_cls = self.model.__class__
            self.model = model_cls.load_from_checkpoint(
                checkpoint_path=checkpoint,
                map_location="cpu",
                **kwargs_user_input,
            )
            self.model.eval()

        self.model.explain_mode = explain
        exported_model_path = self.model.export(
            output_dir=Path(self.work_dir),
            base_name=self._EXPORTED_MODEL_BASE_NAME,
            export_format=export_format,
            precision=export_precision,
            to_exportable_code=export_demo_package,
        )

        self.model.explain_mode = False
        return exported_model_path

    def optimize(
        self,
        checkpoint: PathLike | None = None,
        datamodule: TRAIN_DATALOADERS | OTXDataModule | None = None,
        max_data_subset_size: int | None = None,
        export_demo_package: bool = False,
    ) -> Path:
        r"""Applies NNCF.PTQ to the underlying models (now works only for OV models).

        PTQ performs int-8 quantization on the input model, so the resulting model
        comes in mixed precision (some operations, however, remain in FP32).

        Args:
            checkpoint (str | Path | None, optional): Checkpoint to optimize. Defaults to None.
            datamodule (TRAIN_DATALOADERS | OTXDataModule | None, optional): The data module to use for optimization.
            max_data_subset_size (int | None): The maximum size of the train subset from `datamodule` that would be
            used for model optimization. If not set, NNCF.PTQ will select subset size according to it's
            default settings.
            export_demo_package (bool): Whether to export demo package with optimized models.
            It outputs zip archive with stand-alone demo package.

        Returns:
            Path: path to the optimized model.

        Example:
            >>> engine.optimize(
            ...     checkpoint=<checkpoint/path>,
            ...     datamodule=OTXDataModule(),
            ...     checkpoint=<checkpoint/path>,
            ... )

        CLI Usage:
            1. To optimize a model with IR Model, run
                ```shell
                >>> otx optimize \
                ...     --work_dir <WORK_DIR_PATH, str> \
                ...     --checkpoint <IR_MODEL_WEIGHT_PATH, str>
                ```
            2. To optimize a specific OVModel class with XML, run
                ```shell
                >>> otx optimize \
                ...     --data_root <DATASET_PATH, str> \
                ...     --checkpoint <IR_MODEL_WEIGHT_PATH, str> \
                ...     --model <CONFIG | CLASS_PATH_OR_NAME, OVModel> \
                ...     --model.model_name=<PATH_TO_IR_XML, str>
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

        if not export_demo_package:
            return model.optimize(
                Path(self.work_dir),
                optimize_datamodule,
                ptq_config,
            )

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_model_path = model.optimize(Path(tmp_dir), optimize_datamodule, ptq_config)
            return self.export(
                checkpoint=tmp_model_path,
                export_demo_package=True,
            )

    def explain(
        self,
        checkpoint: PathLike | None = None,
        datamodule: EVAL_DATALOADERS | OTXDataModule | None = None,
        explain_config: ExplainConfig | None = None,
        dump: bool | None = False,
        **kwargs,
    ) -> list | None:
        r"""Run XAI using the specified model and data (test subset).

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
            1. To run XAI with the torch model in work_dir, run
                ```shell
                >>> otx explain \
                ...     --work_dir <WORK_DIR_PATH, str>
                ```
            2. To run XAI using the specified model (torch or IR), run
                ```shell
                >>> otx explain \
                ...     --work_dir <WORK_DIR_PATH, str> \
                ...     --checkpoint <CKPT_PATH, str>
                ```
            3. To run XAI using the configuration, run
                ```shell
                >>> otx explain \
                ...     --config <CONFIG_PATH> --data_root <DATASET_PATH, str> \
                ...     --checkpoint <CKPT_PATH, str>
                ```
        """
        from otx.algo.utils.xai_utils import (
            dump_saliency_maps,
            process_saliency_maps_in_pred_entity,
            set_crop_padded_map_flag,
        )

        model = self.model

        checkpoint = checkpoint if checkpoint is not None else self.checkpoint
        datamodule = datamodule if datamodule is not None else self.datamodule

        is_ir_ckpt = checkpoint is not None and Path(checkpoint).suffix in [".xml", ".onnx"]
        if is_ir_ckpt and not isinstance(model, OVModel):
            datamodule = self._auto_configurator.update_ov_subset_pipeline(datamodule=datamodule, subset="test")
            model = self._auto_configurator.get_ov_model(model_name=str(checkpoint), label_info=datamodule.label_info)

        if checkpoint is not None and not is_ir_ckpt:
            kwargs_user_input: dict[str, Any] = {}
            if self.task == OTXTaskType.ZERO_SHOT_VISUAL_PROMPTING:
                # to update user's custom infer_reference_info_root through cli for zero-shot learning
                # TODO (sungchul): revisit for better solution
                kwargs_user_input.update(infer_reference_info_root=self.model.infer_reference_info_root)

            model_cls = model.__class__
            model = model_cls.load_from_checkpoint(checkpoint_path=checkpoint, **kwargs_user_input)

        if model.label_info != self.datamodule.label_info:
            msg = (
                "To launch a explain pipeline, the label information should be same "
                "between the training and testing datasets. "
                "Please check whether you use the same dataset: "
                f"model.label_info={model.label_info}, "
                f"datamodule.label_info={self.datamodule.label_info}"
            )
            raise ValueError(msg)

        model.explain_mode = True

        self._build_trainer(**kwargs)

        predict_result = self.trainer.predict(
            model=model,
            datamodule=datamodule,
        )

        if explain_config is None:
            explain_config = ExplainConfig()
        explain_config = set_crop_padded_map_flag(explain_config, datamodule)

        predict_result = process_saliency_maps_in_pred_entity(predict_result, explain_config, datamodule.label_info)
        if dump:
            dump_saliency_maps(
                predict_result,
                explain_config,
                datamodule,
                output_dir=Path(self.work_dir),
            )
        model.explain_mode = False
        return predict_result

    def benchmark(
        self,
        checkpoint: PathLike | None = None,
        batch_size: int = 1,
        n_iters: int = 10,
        extended_stats: bool = False,
        print_table: bool = True,
    ) -> dict[str, str]:
        r"""Executes model micro benchmarking on random data.

        Benchmark can provide latency, throughput, number of parameters,
        and theoretical computational complexity with batch size 1.
        The latter two characteristics are available for torch model recipes only.
        Before the measurements, a warm-up is done.

        Args:
            checkpoint (PathLike | None, optional): Path to checkpoint. Optional for torch models. Defaults to None.
            batch_size (int, optional): Batch size for benchmarking. Defaults to 1.
            n_iters (int, optional): Number of iterations to average on. Defaults to 10.
            extended_stats (bool, optional): Flag that enables printing of per module complexity for torch model.
                Defaults to False.
            print_table (bool, optional): Flag that enables printing the benchmark results in a rich table.
                Defaults to True.

        Returns:
            dict[str, str]: a dict with the benchmark results.

        Example:
            >>> engine.benchmark(
            ...     checkpoint=<checkpoint-path>,
            ...     batch_size=1,
            ...     n_iters=20,
            ...     extended_stats=True,
            ... )

        CLI Usage:
            1. To run benchmark by specifying the work_dir where did the training, run
                ```shell
                >>> otx benchmark --work_dir <WORK_DIR_PATH, str>
                ```
            2. To run benchmark by specifying the checkpoint, run
                ```shell
                >>> otx benchmark \
                ...     --work_dir <WORK_DIR_PATH, str> \
                ...     --checkpoint <CKPT_PATH, str>
                ```
            3. To run benchmark using the configuration, launch
                ```shell
                >>> otx benchmark \
                ...     --config <CONFIG_PATH> \
                ...     --data_root <DATASET_PATH, str> \
                ...     --checkpoint <CKPT_PATH, str>
                ```
        """
        checkpoint = checkpoint if checkpoint is not None else self.checkpoint

        if checkpoint is not None:
            is_ir_ckpt = Path(checkpoint).suffix in [".xml"]
            if is_ir_ckpt and not isinstance(self.model, OVModel):
                # create OVModel
                self.model = self._auto_configurator.get_ov_model(
                    model_name=str(checkpoint),
                    label_info=self.datamodule.label_info,
                )

            if not is_ir_ckpt:
                kwargs_user_input: dict[str, Any] = {}
                if self.task == OTXTaskType.ZERO_SHOT_VISUAL_PROMPTING:
                    # to update user's custom infer_reference_info_root through cli for zero-shot learning
                    # TODO (sungchul): revisit for better solution
                    kwargs_user_input.update(infer_reference_info_root=self.model.infer_reference_info_root)

                model_cls = self.model.__class__
                self.model = model_cls.load_from_checkpoint(
                    checkpoint_path=checkpoint,
                    map_location="cpu",
                    **kwargs_user_input,
                )
        elif isinstance(self.model, OVModel):
            msg = "To run benchmark on OV model, checkpoint must be specified."
            raise RuntimeError(msg)

        self.model.eval()

        def dummy_infer(model: OTXModel, batch_size: int = 1) -> float:
            input_batch = model.get_dummy_input(batch_size)
            start = time.perf_counter()
            model.forward(input_batch)
            end = time.perf_counter()
            return end - start

        warmup_iters = max(1, int(n_iters / 10))
        for _ in range(warmup_iters):
            dummy_infer(self.model, batch_size)

        total_time = 0.0
        for _ in range(n_iters):
            total_time += dummy_infer(self.model, batch_size)
        latency = total_time / n_iters
        fps = batch_size / latency

        final_stats = {"latency": f"{latency:.3f} s", "throughput": f"{(fps):.3f} FPS"}

        if not isinstance(self.model, OVModel):
            try:
                from torch.utils.flop_counter import convert_num_with_suffix, get_suffix_str

                input_batch = self.model.get_dummy_input(1)
                model_fwd = lambda: self.model.forward(input_batch)
                depth = 3 if extended_stats else 0
                fwd_flops = measure_flops(self.model.model, model_fwd, print_stats_depth=depth)
                flops_str = convert_num_with_suffix(fwd_flops, get_suffix_str(fwd_flops * 10**3))
                final_stats["complexity"] = flops_str + " MACs"
            except Exception as e:
                logging.warning(f"Failed to complete complexity estimation: {e}")

            params_num = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            params_num_str = convert_num_with_suffix(params_num, get_suffix_str(params_num * 100))
            final_stats["parameters_number"] = params_num_str

        if print_table:
            from rich.console import Console
            from rich.table import Column, Table

            console = Console()
            table_headers = ["Benchmark", "Value"]
            columns = [Column(h, justify="center", style="magenta", width=console.width) for h in table_headers]
            columns[0].style = "cyan"
            table = Table(*columns)
            for name, val in final_stats.items():
                table.add_row(*[f"{name:<20}", f"{val}"])
            console.print(table)

        with (Path(self.work_dir) / "benchmark_report.csv").open("w") as f:
            writer = csv.writer(f)
            writer.writerow(list(final_stats))
            writer.writerow(list(final_stats.values()))

        return final_stats

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

        Returns:
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

        if (datamodule := instantiated_config.get("data")) is None:
            msg = "Cannot instantiate datamodule from config."
            raise ValueError(msg)
        if not isinstance(datamodule, OTXDataModule):
            raise TypeError(datamodule)

        if (model := instantiated_config.get("model")) is None:
            msg = "Cannot instantiate model from config."
            raise ValueError(msg)
        if not isinstance(model, OTXModel):
            raise TypeError(model)

        model.label_info = datamodule.label_info

        return cls(
            work_dir=instantiated_config.get("work_dir", work_dir),
            datamodule=datamodule,
            model=model,
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
                ...     "data.train_subset.batch_size": 2,
                ...     "data.test_subset.subset_name": "TESTING",
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
    def num_devices(self) -> int:
        """Number of devices for Engine use."""
        return self._device.devices

    @num_devices.setter
    def num_devices(self, num_devices: int) -> None:
        """Setter function for multi-gpu."""
        self._device.devices = num_devices
        self._cache.update(devices=self._device.devices)
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
