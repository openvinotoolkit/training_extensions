# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Class definition for base model entity used in getitune."""

# mypy: disable-error-code="arg-type"

from __future__ import annotations

import inspect
import logging
import warnings
from abc import abstractmethod
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Callable, Literal, Sequence

import torch
from lightning import LightningModule, Trainer
from torch import Tensor, nn
from torch.optim.lr_scheduler import ConstantLR
from torch.optim.sgd import SGD
from torchmetrics import Metric, MetricCollection

from getitune import __version__
from getitune.backend.lightning.optimizers.callable import OptimizerCallableSupportAdaptiveBS
from getitune.backend.lightning.schedulers import (
    LinearWarmupScheduler,
    LinearWarmupSchedulerCallable,
    LRSchedulerListCallable,
    SchedulerCallableSupportAdaptiveBS,
)
from getitune.backend.lightning.utils.utils import (
    ensure_callable,
    is_ckpt_for_finetuning,
    remove_state_dict_prefix,
)
from getitune.config.data import TileConfig
from getitune.data.entity.base import (
    BatchLoss,
)
from getitune.data.entity.sample import PredictionBatch, SampleBatch
from getitune.data.entity.tile import TileBatchData
from getitune.metrics import MetricInput, NullMetricCallable
from getitune.types.export import ExportFormat, TaskLevelExportParameters
from getitune.types.label import LabelInfo, LabelInfoTypes
from getitune.types.precision import Precision
from getitune.types.task import TaskType

if TYPE_CHECKING:
    from pathlib import Path

    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
    from lightning.pytorch.utilities.types import LRSchedulerTypeUnion
    from torch.optim.lr_scheduler import LRScheduler
    from torch.optim.optimizer import Optimizer, params_t

    from getitune.backend.lightning.exporter.base import ModelExporter
    from getitune.config.data import IntensityConfig
    from getitune.data.module import DataModule
    from getitune.metrics import MetricCallable
    from getitune.types import PathLike

logger = logging.getLogger()


@dataclass
class DataInputParams:
    """Parameters of the input data such as input size, mean, and std.

    Attributes:
        input_size: Spatial dimensions (H, W) expected by the model.
        mean: Per-channel mean for normalization.
        std: Per-channel std for normalization.
        intensity_config: Optional intensity mapping configuration for
            high-bit-depth inputs (uint16, thermal, medical, etc.).
            When present, the exporter embeds these parameters into the
            exported model's ``rt_info`` / ONNX metadata so that ModelAPI
            can reconstruct the correct preprocessing at inference time.

    Note:
        Mean and std values are written directly to the exported model metadata
        (``model_info/mean_values`` and ``model_info/scale_values``). ModelAPI
        applies ``(input - mean) / scale`` after converting the input to float.

        - Models that use ImageNet normalization on [0,1] images (e.g., DEIM, ViT,
          YOLOX-tiny) should use 0-1 range values: mean=(0.485, 0.456, 0.406),
          std=(0.229, 0.224, 0.225).
        - Models that expect [0,255] input but receive [0,1] from the intensity
          pipeline (e.g., YOLOX s/l/x) encode the x*255 scaling in std:
          mean=(0, 0, 0), std=(1/255, 1/255, 1/255). ModelAPI then computes
          (x - 0) / (1/255) = x * 255.
    """

    input_size: tuple[int, int]
    mean: tuple[float, float, float] | None
    std: tuple[float, float, float] | None
    intensity_config: IntensityConfig | None = None

    def as_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result: dict[str, Any] = {"input_size": self.input_size, "mean": self.mean, "std": self.std}
        if self.intensity_config is not None:
            result["intensity_config"] = asdict(self.intensity_config)
        return result

    def as_ncwh(self, batch_size: int = 1) -> tuple[int, int, int, int]:
        """Convert input_size to NCWH format."""
        if self.input_size is not None:
            return (batch_size, 3, *self.input_size)

        msg = "input_size should not be None."
        raise ValueError(msg)


def _default_optimizer_callable(params: params_t) -> Optimizer:
    return SGD(params=params, lr=0.01)


def _default_scheduler_callable(
    optimizer: Optimizer,
    interval: Literal["epoch", "step"] = "epoch",
    **kwargs,
) -> LRScheduler:
    scheduler = ConstantLR(optimizer=optimizer, **kwargs)
    # NOTE: "interval" attribute should be set to configure the scheduler's step interval correctly
    scheduler.interval = interval
    return scheduler


DefaultOptimizerCallable = _default_optimizer_callable
DefaultSchedulerCallable = _default_scheduler_callable


class LightningModel(LightningModule):
    """Base class for the models used in getitune.

    This class is a subclass of `LightningModule`. It is not intended to be used directly.

    Args:
        label_info (LabelInfoTypes | int | Sequence): Information about the labels used in the model.
            If `int` is given, label info will be constructed from number of classes,
            if `Sequence` is given, label info will be constructed from the sequence of label names.
        model_name (str, optional): Name of the model. Defaults to "LightningModel".
        optimizer (OptimizerCallable, optional): Optimizer callable. Defaults to DefaultOptimizerCallable.
        scheduler (LRSchedulerCallable | LRSchedulerListCallable, optional): Scheduler callable.
            Defaults to DefaultSchedulerCallable.
        metric (MetricCallable, optional): Metric callable. Defaults to NullMetricCallable.
        torch_compile (bool, optional): Whether to use torch compile. Defaults to False.
        tile_config (TileConfig | dict, optional): Configuration for tiling. Defaults to TileConfig(enable_tiler=False).
        data_input_params (DataInputParams | dict | None, optional): Parameters for image preprocessing.
            This parameter contains image input size, mean, and std, that is used to preprocess the input image.
            If None is given, default parameters for the specific model will be used.
            In most cases you don't need to set this parameter unless you change the image size or pretrained weights.
            Defaults to None.

    Attributes:
        explain_mode: If true, `self.predict_step()` will produce a XAI output as well
        input_size_multiplier (int):
            multiplier value for input size a model requires. If input_size isn't multiple of this value,
            error is raised.
    """

    _OPTIMIZED_MODEL_BASE_NAME: str = "optimized_model"
    input_size_multiplier: int = 1

    def __init__(
        self,
        label_info: LabelInfoTypes | int | Sequence,
        data_input_params: DataInputParams | dict | None = None,
        model_name: str = "LightningModel",
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = NullMetricCallable,
        torch_compile: bool = False,
        tile_config: TileConfig | dict = TileConfig(enable_tiler=False),
        pretrained: bool = True,
        pretrained_weights: PathLike | None = None,
    ) -> None:
        """Initialize the base model with the given parameters.

        Args:
            label_info (LabelInfoTypes | int | Sequence): Information about the labels used in the model.
                If `int` is given, label info will be constructed from number of classes,
                if `Sequence` is given, label info will be constructed from the sequence of label names.
            data_input_params (DataInputParams | dict | None, optional): Parameters for image preprocessing.
                This parameter contains image input size, mean, and std, that is used to preprocess the input image.
                If None is given, default parameters for the specific model will be used.
                Defaults to None.
            model_name (str, optional): Name of the model. Defaults to "LightningModel".
            optimizer (OptimizerCallable, optional): Callable for the optimizer. Defaults to DefaultOptimizerCallable.
            scheduler (LRSchedulerCallable | LRSchedulerListCallable): Callable for the learning rate scheduler.
                Defaults to DefaultSchedulerCallable.
            metric (MetricCallable, optional): Callable for the metric. Defaults to NullMetricCallable.
            torch_compile (bool, optional): Flag to indicate if torch.compile should be used. Defaults to False.
            tile_config (TileConfig, optional): Configuration for tiling. Defaults to TileConfig(enable_tiler=False).
            pretrained (bool, optional): Whether to use pretrained weights. Defaults to True.
            pretrained_weights (PathLike | None, optional): Path to the pretrained weights file. When None is passed,
                the default pretrained weights will be utilized for fine-tuning. Defaults to None.

        """
        super().__init__()

        self._label_info = self._dispatch_label_info(label_info)
        self.model_name = model_name
        self.data_input_params = self._configure_preprocessing_params(data_input_params)
        self.model = self._create_model()
        if pretrained:
            self.load_pretrained(pretrained_weights)
        self.optimizer_callable = ensure_callable(optimizer)
        self.scheduler_callable = ensure_callable(scheduler)
        self.metric_callable = ensure_callable(metric)
        self.torch_compile = torch_compile
        self._explain_mode = False

        # NOTE: To guarantee immutability of the default value
        if isinstance(tile_config, dict):
            tile_config = TileConfig(**tile_config)
        self._tile_config = tile_config.clone()

        self.save_hyperparameters(
            logger=False,
            ignore=[
                "optimizer",
                "scheduler",
                "metric",
                "label_info",
                "tile_config",
                "data_input_params",
                "pretrained_weights",
            ],
        )

    def training_step(self, batch: SampleBatch, batch_idx: int) -> Tensor:
        """Step for model training."""
        train_loss = self.forward(inputs=batch)
        if train_loss is None:
            msg = "Loss is None."
            raise ValueError(msg)

        if isinstance(train_loss, Tensor):
            self.log(
                "train/loss",
                train_loss,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
            )
            return train_loss
        if isinstance(train_loss, dict):
            for k, v in train_loss.items():
                self.log(
                    f"train/{k}",
                    v,
                    on_step=True,
                    on_epoch=False,
                    prog_bar=True,
                )

            total_train_loss = train_loss.get("total_loss", sum(train_loss.values()))
            self.log(
                "train/total_loss",
                total_train_loss,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
            )
            return total_train_loss

        raise TypeError(train_loss)

    def validation_step(self, batch: SampleBatch, batch_idx: int) -> PredictionBatch:
        """Perform a single test step on a batch of data from the test set.

        Args:
            batch: A batch of data containing the input tensor of images and target labels.
            batch_idx: The index of the current batch.

        Returns:
            PredictionBatch: The prediction results for the batch.

        Raises:
            TypeError: If predictions are of type BatchLoss or if metric inputs
                    have an unsupported type.

        Note:
            Updates test metrics based on the prediction results and batch data.
            Handles both single dictionary and list of dictionaries for metric inputs.
        """
        preds = self.forward(inputs=batch)

        if isinstance(preds, BatchLoss):
            raise TypeError(preds)

        metric_inputs = self._convert_pred_entity_to_compute_metric(preds, batch)

        if isinstance(metric_inputs, dict):
            self.metric.update(**metric_inputs)
            return preds

        if isinstance(metric_inputs, list) and all(isinstance(inp, dict) for inp in metric_inputs):
            for inp in metric_inputs:
                self.metric.update(**inp)
            return preds

        raise TypeError(metric_inputs)

    def test_step(self, batch: SampleBatch, batch_idx: int) -> PredictionBatch:
        """Lightning hook called at the beginning of fit, validate, test, or predict stages.

        This hook is used for dynamic model building or model adjustments. It is called
        on every process when using Distributed Data Parallel (DDP).

        Args:
            stage: The current stage, either "fit", "validate", "test", or "predict".

        Note:
            When torch_compile is enabled and stage is "fit", compiles the model for
            optimized performance with appropriate logging level adjustments.
        """
        preds = self.forward(inputs=batch)

        if isinstance(preds, BatchLoss):
            raise TypeError(preds)

        metric_inputs = self._convert_pred_entity_to_compute_metric(preds, batch)

        if isinstance(metric_inputs, dict):
            self.metric.update(**metric_inputs)
            return preds

        if isinstance(metric_inputs, list) and all(isinstance(inp, dict) for inp in metric_inputs):
            for inp in metric_inputs:
                self.metric.update(**inp)
            return preds

        raise TypeError(metric_inputs)

    def predict_step(
        self,
        batch: SampleBatch | TileBatchData,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> PredictionBatch:
        """Step function called during PyTorch Lightning Trainer's predict."""
        if self.explain_mode:
            return self.forward_explain(inputs=batch)

        outputs = self.forward(inputs=batch)

        if isinstance(outputs, BatchLoss):
            raise TypeError(outputs)

        return outputs

    def on_validation_start(self) -> None:
        """Called at the beginning of validation."""
        self.configure_metric()

    def on_test_start(self) -> None:
        """Called at the beginning of testing."""
        self.configure_metric()

    def on_validation_epoch_start(self) -> None:
        """Callback triggered when the validation epoch starts."""
        self.metric.reset()

    def on_test_epoch_start(self) -> None:
        """Callback triggered when the test epoch starts."""
        self.metric.reset()

    def on_validation_epoch_end(self) -> None:
        """Callback triggered when the validation epoch ends."""
        self._log_metrics(self.metric, "val")

    def on_test_epoch_end(self) -> None:
        """Callback triggered when the test epoch ends."""
        self._log_metrics(self.metric, "test")

    def setup(self, stage: str) -> None:
        """Lightning hook called at the beginning of fit, validate, test, or predict stages.

        This hook is used for dynamic model building or model adjustments. It is called
        on every process when using Distributed Data Parallel (DDP).

        Args:
            stage: The current stage, either "fit", "validate", "test", or "predict".

        Note:
            When torch_compile is enabled and stage is "fit", compiles the model for
            optimized performance with appropriate logging level adjustments.
        """
        if self.torch_compile and stage == "fit":
            # Set the log_level of this to error due to the numerous warning messages from compile.
            torch._logging.set_logs(dynamo=logging.ERROR)  # noqa: SLF001
            self.model = torch.compile(self.model)
            warnings.warn(
                (
                    "torch model compile has been applied. It may be slower than usual because "
                    "it builds the graph in the initial training."
                ),
                stacklevel=1,
            )

    def configure_optimizers(self) -> tuple[list[Optimizer], list[dict[str, Any]]]:
        """Configure an optimizer and learning-rate schedulers.

        Configure an optimizer and learning-rate schedulers
        from the given optimizer and scheduler or scheduler list callable in the constructor.
        Generally, there is two lr schedulers. One is for a linear warmup scheduler and
        the other is the main scheduler working after the warmup period.

        Returns:
            Two list. The former is a list that contains an optimizer
            The latter is a list of lr scheduler configs which has a dictionary format.
        """
        optimizer = self.optimizer_callable(self.parameters())
        schedulers = self.scheduler_callable(optimizer)

        def ensure_list(item: Any) -> list:  # noqa: ANN401
            return item if isinstance(item, list) else [item]

        lr_scheduler_configs = []
        for scheduler in ensure_list(schedulers):
            lr_scheduler_config = {"scheduler": scheduler}
            if hasattr(scheduler, "interval"):
                lr_scheduler_config["interval"] = scheduler.interval
            if hasattr(scheduler, "monitor"):
                lr_scheduler_config["monitor"] = scheduler.monitor
            lr_scheduler_configs.append(lr_scheduler_config)

        return [optimizer], lr_scheduler_configs

    def configure_metric(self) -> None:
        """Configure the metric."""
        if not callable(self.metric_callable):
            raise TypeError(self.metric_callable)

        metric = self.metric_callable(self.label_info)

        if not isinstance(metric, (Metric, MetricCollection)):
            msg = "Metric should be the instance of `torchmetrics.Metric` or `torchmetrics.MetricCollection`."
            raise TypeError(msg, metric)

        self._metric = metric.to(self.device)

    @property
    def metric(self) -> Metric | MetricCollection:
        """Metric module for this getitune model."""
        return self._metric

    @abstractmethod
    def _convert_pred_entity_to_compute_metric(
        self,
        preds: PredictionBatch,
        inputs: SampleBatch,
    ) -> MetricInput:
        """Convert given inputs to a Python dictionary for the metric computation."""
        raise NotImplementedError

    def _log_metrics(self, meter: Metric, key: Literal["val", "test"], **compute_kwargs) -> None:
        sig = inspect.signature(meter.compute)
        filtered_kwargs = {key: value for key, value in compute_kwargs.items() if key in sig.parameters}
        if removed_kwargs := set(compute_kwargs.keys()).difference(filtered_kwargs.keys()):
            msg = f"These keyword arguments are removed since they are not in the function signature: {removed_kwargs}"
            logger.debug(msg)

        results: dict[str, Tensor] = meter.compute(**filtered_kwargs)

        if not isinstance(results, dict):
            raise TypeError(results)

        if not results:
            msg = f"{meter} has no data to compute metric or there is an error computing metric"
            raise RuntimeError(msg)

        # Known auxiliary (non-scalar) entries emitted by torchmetrics'
        # MeanAveragePrecision / classification metrics. These are intentionally
        # non-scalar and do not need to be logged as Lightning scalars.
        _known_non_scalar_keys = {
            "classes",
            "map_per_class",
            "mar_100_per_class",
            "ious",
        }

        for name, value in results.items():
            log_metric_name = f"{key}/{name}"

            if not isinstance(value, Tensor) or value.numel() != 1:
                if name not in _known_non_scalar_keys:
                    msg = f"Log metric name={log_metric_name} is not a scalar tensor. Skip logging it."
                    warnings.warn(msg, stacklevel=1)
                continue

            self.log(log_metric_name, value.to(self.device), sync_dist=True, prog_bar=True)

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Callback on saving checkpoint."""
        if self.torch_compile:
            # If torch_compile is True, a prefix key named _orig_mod. is added to the state_dict. Remove this.
            compiled_state_dict = checkpoint["state_dict"]
            checkpoint["state_dict"] = remove_state_dict_prefix(compiled_state_dict, "_orig_mod.")
        super().on_save_checkpoint(checkpoint)
        checkpoint["hyper_parameters"]["label_info"] = asdict(self.label_info)
        checkpoint["getitune_version"] = __version__
        checkpoint["hyper_parameters"]["tile_config"] = asdict(self.tile_config)
        checkpoint["hyper_parameters"]["data_input_params"] = asdict(self.data_input_params)
        checkpoint.pop("datamodule_hparams_name", None)
        checkpoint.pop(
            "datamodule_hyper_parameters",
            None,
        )  # Remove datamodule_hyper_parameters to prevent storing getitune classes

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Callback on loading checkpoint."""
        super().on_load_checkpoint(checkpoint)
        hyper_parameters = checkpoint.get("hyper_parameters")
        if hyper_parameters:
            if ckpt_label_info := hyper_parameters.get("label_info"):
                self._label_info = self._dispatch_label_info(ckpt_label_info)
            if ckpt_tile_config := hyper_parameters.get("tile_config"):
                if isinstance(ckpt_tile_config, dict):
                    ckpt_tile_config = TileConfig(**ckpt_tile_config)
                self.tile_config = ckpt_tile_config

    def load_state_dict_incrementally(self, ckpt: dict[str, Any], *args, **kwargs) -> None:
        """Load state dict incrementally."""
        ckpt_label_info: LabelInfo | None = ckpt.get("hyper_parameters", {}).get("label_info")

        if ckpt_label_info is None:
            msg = "Checkpoint should have `label_info`."
            raise ValueError(msg, ckpt_label_info)

        ckpt_label_info = self._dispatch_label_info(ckpt_label_info)

        if not hasattr(ckpt_label_info, "label_ids"):
            msg = "Loading checkpoint from getitune < 2.2.1, label_ids are assigned automatically"
            logger.info(msg)
            ckpt_label_info.label_ids = [str(i) for i, _ in enumerate(ckpt_label_info.label_names)]

        if not set(ckpt_label_info.label_names).isdisjoint(self.label_info.label_names):
            msg = (
                "Load model state dictionary incrementally: "
                f"Label info from checkpoint: {ckpt_label_info} -> "
                f"Label info from training data: {self.label_info}"
            )
            logger.info(msg)
            self.register_load_state_dict_pre_hook(
                self.label_info.label_names,
                ckpt_label_info.label_names,
            )

        # Model weights
        state_dict: dict[str, Any] = ckpt.get("state_dict", {})

        if state_dict is None or state_dict == {}:
            msg = "Checkpoint should have `state_dict`."
            raise ValueError(msg, state_dict)

        self.load_state_dict(state_dict, *args, **kwargs)

    def load_state_dict(self, ckpt: dict[str, Any], *args, **kwargs) -> None:
        """Load state dictionary from checkpoint state dictionary.

        It successfully loads the checkpoint from getitune v1.x and for finetune and for resume.

        If checkpoint's label_info and LightningModel's label_info are different,
        load_state_pre_hook for smart weight loading will be registered.
        """
        if is_ckpt_for_finetuning(ckpt):
            self.on_load_checkpoint(ckpt)
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
        return super().load_state_dict(state_dict, *args, **kwargs)

    @property
    def label_info(self) -> LabelInfo:
        """Get this model label information."""
        return self._label_info

    @label_info.setter
    def label_info(self, label_info: LabelInfoTypes) -> None:
        """Set this model label information."""
        self._set_label_info(label_info)

    def _set_label_info(self, label_info: LabelInfoTypes) -> None:
        """Actual implementation for set this model label information.

        Derived classes should override this function.
        """
        msg = (
            "Assign new label_info to the model. "
            "It is usually not recommended. "
            "Please create a new model instance by giving label_info to its initializer "
            "such as `LightningModel(label_info=label_info, ...)`."
        )
        logger.warning(msg, stacklevel=0)

        new_label_info = self._dispatch_label_info(label_info)

        old_num_classes = self._label_info.num_classes
        new_num_classes = new_label_info.num_classes

        if old_num_classes != new_num_classes:
            msg = (
                f"Given LabelInfo has the different number of classes "
                f"({old_num_classes}!={new_num_classes}). "
                "The model prediction layer is reset to the new number of classes "
                f"(={new_num_classes})."
            )
            logger.warning(msg, stacklevel=0)
            self._reset_prediction_layer(num_classes=new_label_info.num_classes)

        self._label_info = new_label_info

    @property
    @abstractmethod
    def _default_preprocessing_params(self) -> DataInputParams | dict[str, DataInputParams]:
        """Parameters for image preprocessing.

        Each model architecture must implement this property, returning a DataInputParams
        containing the image input size, mean, and std, that is used to preprocess the input image.
        """

    @property
    def num_classes(self) -> int:
        """Returns model's number of classes. Can be redefined at the model's level."""
        return self.label_info.num_classes

    @property
    def explain_mode(self) -> bool:
        """Get model explain mode."""
        return self._explain_mode

    @explain_mode.setter
    def explain_mode(self, explain_mode: bool) -> None:
        """Set model explain mode."""
        self._explain_mode = explain_mode

    @abstractmethod
    def _create_model(self, num_classes: int | None = None) -> nn.Module:
        """Create a PyTorch model for this class."""

    def _customize_inputs(self, inputs: SampleBatch) -> dict[str, Any]:
        """Customize getitune input batch data entity if needed for your model."""
        raise NotImplementedError

    def _customize_outputs(
        self,
        outputs: Any,  # noqa: ANN401
        inputs: SampleBatch,
    ) -> PredictionBatch | BatchLoss:
        """Customize getitune output batch data entity if needed for model."""
        raise NotImplementedError

    def forward(
        self,
        inputs: SampleBatch | Tensor,
    ) -> PredictionBatch | BatchLoss | Tensor:
        """Model forward function."""
        # Simple forward
        if isinstance(inputs, Tensor):
            return self.forward_for_tracing(inputs)

        # If customize_inputs is overridden
        if isinstance(inputs, TileBatchData):
            return self.forward_tiles(inputs)

        outputs = (
            self.model(**self._customize_inputs(inputs))
            if self._customize_inputs != LightningModel._customize_inputs
            else self.model(inputs)
        )

        return (
            self._customize_outputs(outputs, inputs)
            if self._customize_outputs != LightningModel._customize_outputs
            else outputs
        )

    def forward_explain(self, inputs: SampleBatch) -> PredictionBatch:
        """Model forward explain function."""
        msg = "Derived model class should implement this class to support the explain pipeline."
        raise NotImplementedError(msg)

    def forward_for_tracing(self, *args, **kwargs) -> Tensor | dict[str, Tensor]:
        """Model forward function used for the model tracing during model exportation."""
        msg = (
            "Derived model class should implement this class to support the export pipeline. "
            "If it wants to use `getitune.core.exporter.native.LightningModelExporter`."
        )
        raise NotImplementedError(msg)

    def get_explain_fn(self) -> Callable:
        """Returns explain function."""
        raise NotImplementedError

    def forward_tiles(
        self,
        inputs: TileBatchData,
    ) -> PredictionBatch | BatchLoss:
        """Model forward function for tile task."""
        raise NotImplementedError

    def register_load_state_dict_pre_hook(self, model_classes: list[str], ckpt_classes: list[str]) -> None:
        """Register load_state_dict_pre_hook.

        Args:
            model_classes (list[str]): Class names from training data.
            ckpt_classes (list[str]): Class names from checkpoint state dictionary.
        """
        self.model_classes = model_classes
        self.ckpt_classes = ckpt_classes
        self._register_load_state_dict_pre_hook(self.load_state_dict_pre_hook)

    def load_state_dict_pre_hook(self, state_dict: dict[str, torch.Tensor], prefix: str, *args, **kwargs) -> None:
        """Modify input state_dict according to class name matching before weight loading."""
        model2ckpt = self.map_class_names(self.model_classes, self.ckpt_classes)

        for param_name in self._identify_classification_layers():
            model_param = self.state_dict()[param_name].clone()
            ckpt_param = state_dict[prefix + param_name]
            for model_dst, ckpt_dst in enumerate(model2ckpt):
                if ckpt_dst >= 0:
                    model_param[model_dst : model_dst + 1].copy_(
                        ckpt_param[ckpt_dst : ckpt_dst + 1],
                    )

            # Replace checkpoint weight by mixed weights
            state_dict[prefix + param_name] = model_param

    def _identify_classification_layers(self, prefix: str = "model.") -> list[str]:
        """Simple identification of the classification layers."""
        # identify classification layers
        sample_model_dict = self._create_model(num_classes=3).state_dict()
        incremental_model_dict = self._create_model(num_classes=4).state_dict()
        # iterate over the model dict and compare the shapes.
        # Add the key to the list if the shapes are different
        return [
            prefix + key
            for key in sample_model_dict
            if sample_model_dict[key].shape != incremental_model_dict[key].shape
        ]

    @staticmethod
    def map_class_names(src_classes: list[str], dst_classes: list[str]) -> list[int]:
        """Computes src to dst index mapping.

        src2dst[src_idx] = dst_idx
        #  according to class name matching, -1 for non-matched ones
        assert(len(src2dst) == len(src_classes))
        ex)
          src_classes = ['person', 'car', 'tree']
          dst_classes = ['tree', 'person', 'sky', 'ball']
          -> Returns src2dst = [1, -1, 0]
        """
        src2dst = []
        for src_class in src_classes:
            if src_class in dst_classes:
                src2dst.append(dst_classes.index(src_class))
            else:
                src2dst.append(-1)
        return src2dst

    def optimize(self, output_dir: Path, data_module: DataModule, ptq_config: dict[str, Any] | None = None) -> Path:
        """Runs quantization of the model with NNCF.PTQ on the passed data. Works only for OpenVINO models.

        PTQ performs int-8 quantization on the input model, so the resulting model
        comes in mixed precision (some operations, however, remain in FP32).

        Args:
            output_dir (Path): working directory to save the optimized model.
            data_module (DataModule): dataset for calibration of quantized layers.
            ptq_config (dict[str, Any] | None): config for NNCF.PTQ.

        Returns:
            Path: path to the resulting optimized OpenVINO model.
        """
        msg = "Optimization is not implemented for torch models"
        raise NotImplementedError(msg)

    def _move_model_tensors_to_device_and_dtype(self, device: str, dtype: torch.dtype | None = None) -> nn.Module:
        """Move model and all cached tensors to the specified device and dtype.

        This base implementation simply moves the model to the device.
        Subclasses can override this to handle model-specific cached tensors
        (like anchors in detection models) that are not moved by model.to().

        Args:
            device: Target device ('cpu', 'cuda', etc.)
            dtype: Target dtype for floating-point tensors. If None, dtype is not changed.
                   Note: Integer tensors (used for indexing) should not be converted.

        Returns:
            The model moved to the specified device.
        """
        model = self.model.to(device)
        if dtype is not None:
            model = model.to(dtype)
        return model

    def export(
        self,
        output_dir: Path,
        base_name: str,
        export_format: ExportFormat,
        precision: Precision = Precision.FP32,
    ) -> Path:
        """Export this model to the specified output directory.

        Args:
            output_dir (Path): directory for saving the exported model
            base_name: (str): base name for the exported model file. Extension is defined by the target export format
            export_format (ExportFormat): format of the output model
            precision (Precision): precision of the output model

        Returns:
            Path: path to the exported model.
        """
        mode = self.training
        self.eval()
        # Move model to CPU with FP32 for export to avoid device/dtype mismatch with TorchScript tracing
        # AMP training may leave internal tensors in FP16, which breaks TorchScript check_trace
        original_device = next(self.model.parameters()).device
        original_dtype = next(self.model.parameters()).dtype
        self.model = self._move_model_tensors_to_device_and_dtype("cpu", torch.float32)

        orig_forward = self.forward
        orig_trainer = self._trainer  # type: ignore[has-type]
        try:
            if self._trainer is None:  # type: ignore[has-type]
                self._trainer = Trainer()
            self.forward = self.forward_for_tracing  # type: ignore[method-assign, assignment]
            return self._exporter.export(
                self,
                output_dir,
                base_name,
                export_format,
                precision,
            )
        finally:
            self.train(mode)
            self.forward = orig_forward  # type: ignore[method-assign]
            self._trainer = orig_trainer
            # Restore model to original device and dtype
            self.model = self._move_model_tensors_to_device_and_dtype(str(original_device), original_dtype)

    @property
    def _exporter(self) -> ModelExporter:
        """Defines exporter of the model. Should be overridden in subclasses."""
        msg = (
            "To export this LightningModel, you should implement an appropriate exporter for it. "
            "You can try to reuse ones provided in `getitune.core.exporter.*`."
        )
        raise NotImplementedError(msg)

    @property
    def _export_parameters(self) -> TaskLevelExportParameters:
        """Defines export parameters sharable at a task level.

        To export LightningModel which is compatible with ModelAPI,
        you should define an appropriate export parameters for each task.
        This property is usually defined at the task level classes defined in `getitune.core.model.*`.
        Please refer to `TaskLevelExportParameters` for more details.

        Returns:
            Collection of exporter parameters that can be defined at a task level.

        Examples:
            This example shows how this property is used at the new model development

            ```python

            class MyDetectionModel(LightningDetectionModel):
                ...

                @property
                def _exporter(self) -> ModelExporter:
                    # `self._export_parameters` defined at `LightningDetectionModel`
                    # You can redefine it `MyDetectionModel` if you need
                    return ModelExporter(
                        task_level_export_parameters=self._export_parameters,
                        ...
                    )
            ```
        """
        return TaskLevelExportParameters(
            model_type="null",
            task_type="null",
            model_name=self.model_name,
            label_info=self.label_info,
            optimization_config=self._optimization_config,
        )

    def _reset_prediction_layer(self, num_classes: int) -> None:
        """Reset its prediction layer with a given number of classes.

        Args:
            num_classes: Number of classes
        """
        raise NotImplementedError

    @property
    def _optimization_config(self) -> dict[str, str]:
        return {}

    def lr_scheduler_step(self, scheduler: LRSchedulerTypeUnion, metric: Tensor) -> None:
        """It is required to prioritize the warmup lr scheduler than other lr scheduler during a warmup period.

        It will ignore other lr scheduler's stepping if the warmup scheduler is currently activated.
        """
        warmup_schedulers = [
            config.scheduler
            for config in self.trainer.lr_scheduler_configs
            if isinstance(config.scheduler, LinearWarmupScheduler)
        ]

        if not warmup_schedulers:
            # There is no warmup scheduler
            return super().lr_scheduler_step(scheduler=scheduler, metric=metric)

        if len(warmup_schedulers) != 1:
            msg = "No more than one warmup schedulers coexist."
            raise RuntimeError(msg)

        warmup_scheduler = next(iter(warmup_schedulers))

        if scheduler != warmup_scheduler and warmup_scheduler.activated:
            msg = (
                "Warmup lr scheduler is currently activated. "
                "Ignore other schedulers until the warmup lr scheduler is finished"
            )
            logger.debug(msg)
            return None

        return super().lr_scheduler_step(scheduler=scheduler, metric=metric)

    def patch_optimizer_and_scheduler_for_adaptive_bs(self) -> None:
        """Patch optimizer and scheduler for adaptive batch size.

        This is inplace function changing inner states (`optimizer_callable` and `scheduler_callable`).
        Both will be changed to be picklable. In addition, `optimizer_callable` is changed
        to make its hyperparameters gettable.
        """
        if not isinstance(self.optimizer_callable, OptimizerCallableSupportAdaptiveBS):
            self.optimizer_callable = OptimizerCallableSupportAdaptiveBS.from_callable(self.optimizer_callable)

        if not isinstance(self.scheduler_callable, SchedulerCallableSupportAdaptiveBS) and not isinstance(
            self.scheduler_callable,
            LinearWarmupSchedulerCallable,  # LinearWarmupSchedulerCallable natively supports adaptive batch size
        ):
            self.scheduler_callable = SchedulerCallableSupportAdaptiveBS.from_callable(self.scheduler_callable)

    @property
    def tile_config(self) -> TileConfig:
        """Get tiling configurations."""
        return self._tile_config

    @tile_config.setter
    def tile_config(self, tile_config: TileConfig) -> None:
        """Set tiling configurations."""
        msg = (
            "Assign new tile_config to the model. "
            "It is usually not recommended. "
            "Please create a new model instance by giving tile_config to its initializer "
            "such as `LightningModel(..., tile_config=tile_config)`."
        )
        logger.warning(msg, stacklevel=0)

        self._tile_config = tile_config

    def get_dummy_input(self, batch_size: int = 1) -> SampleBatch:
        """Generates a dummy input, suitable for launching forward() on it.

        Args:
            batch_size (int, optional): number of elements in a dummy input sequence. Defaults to 1.

        Returns:
            TorchDataBatch: A batch containing randomly generated inference data.
        """
        raise NotImplementedError

    @staticmethod
    def _dispatch_label_info(label_info: LabelInfoTypes) -> LabelInfo:
        if isinstance(label_info, dict):
            if "label_ids" not in label_info:
                # NOTE: This is for backward compatibility
                label_info["label_ids"] = label_info["label_names"]
            return LabelInfo(**label_info)
        if isinstance(label_info, int):
            return LabelInfo.from_num_classes(num_classes=label_info)
        if isinstance(label_info, (list, tuple)) and all(isinstance(name, str) for name in label_info):
            return LabelInfo(
                label_names=label_info,
                label_groups=[label_info],
                label_ids=[str(i) for i in range(len(label_info))],
            )
        if isinstance(label_info, LabelInfo):
            if not hasattr(label_info, "label_ids"):
                # NOTE: This is for backward compatibility
                label_info.label_ids = label_info.label_names
            return label_info

        raise TypeError(label_info)

    def _configure_preprocessing_params(
        self,
        preprocessing_params: DataInputParams | dict | None = None,
    ) -> DataInputParams:
        """Check the validity of the preprocessing parameters."""
        default = (
            self._default_preprocessing_params[self.model_name]
            if isinstance(self._default_preprocessing_params, dict)
            else self._default_preprocessing_params
        )

        if isinstance(preprocessing_params, dict):
            # Merge with model defaults for any missing keys so callers can pass
            # a partial dict (e.g. only input_size) without knowing mean/std upfront.
            intensity_cfg = preprocessing_params.get("intensity_config")
            if isinstance(intensity_cfg, dict):
                from getitune.config.data import IntensityConfig

                intensity_cfg = IntensityConfig(**intensity_cfg)
            data_input_params = DataInputParams(
                input_size=preprocessing_params.get("input_size")
                if preprocessing_params.get("input_size") is not None
                else default.input_size,
                mean=preprocessing_params.get("mean") if preprocessing_params.get("mean") is not None else default.mean,
                std=preprocessing_params.get("std") if preprocessing_params.get("std") is not None else default.std,
                intensity_config=intensity_cfg,
            )
        elif isinstance(preprocessing_params, DataInputParams):
            data_input_params = preprocessing_params
        elif preprocessing_params is None:
            data_input_params = default
        else:
            msg = (
                f"preprocessing_params should be either dict or DataInputParams, "
                f"but got {type(preprocessing_params)} instead."
            )
            raise TypeError(msg)

        # Validate
        if data_input_params.mean is None:
            msg = "Mean must be provided (either explicitly or via model defaults)."
            raise ValueError(msg)
        if data_input_params.std is None:
            msg = "Std must be provided (either explicitly or via model defaults)."
            raise ValueError(msg)

        if not (len(data_input_params.mean) == 3 and all(isinstance(m, float) for m in data_input_params.mean)):
            msg = f"Mean should be a tuple of 3 float values, but got {data_input_params.mean} instead."
            raise ValueError(msg)
        if not (len(data_input_params.std) == 3 and all(isinstance(s, float) for s in data_input_params.std)):
            msg = f"Std should be a tuple of 3 float values, but got {data_input_params.std} instead."
            raise ValueError(msg)

        if not all(m >= 0 for m in data_input_params.mean):
            msg = f"Mean values should be non-negative, but got {data_input_params.mean} instead."
            raise ValueError(msg)
        if not all(s > 0 for s in data_input_params.std):
            msg = f"Std values should be positive, but got {data_input_params.std} instead."
            raise ValueError(msg)

        if data_input_params.input_size is not None and (
            data_input_params.input_size[0] % self.input_size_multiplier != 0
            or data_input_params.input_size[1] % self.input_size_multiplier != 0
        ):
            msg = (
                f"Input size should be a multiple of {self.input_size_multiplier}, "
                f"but got {data_input_params.input_size} instead."
            )
            raise ValueError(msg)

        return data_input_params

    @property
    @abstractmethod
    def task(self) -> TaskType:
        """Get  task type."""

    def load_pretrained(self, weights: PathLike | None = None) -> None:
        """Load architecture-specific pretrained weights into ``self.model``.

        Distinct from full getitune checkpoints (resume/fine-tune): this seeds the
        freshly-built architecture. A single combined file covers the whole model
        (backbone + neck + head). When ``weights`` is None, models load their own
        default source. Base implementation is a no-op: if models do not have pretrained
        weights it should work.
        """
