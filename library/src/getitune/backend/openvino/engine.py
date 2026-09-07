# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""OpenVINO engine."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import defusedxml.ElementTree as Elet
import numpy as np
import onnx
import torch
from lightning.pytorch.loggers import CSVLogger
from model_api.adapters.utils import load_parameters_from_onnx
from rich.progress import Progress

from getitune.backend.openvino.models import OVModel
from getitune.config.explain import ExplainConfig
from getitune.data.entity.base import ImageInfo
from getitune.data.entity.sample import SampleBatch
from getitune.data.module import DataModule
from getitune.engine import Engine
from getitune.tools.auto_configurator import AutoConfigurator
from getitune.types import PathLike, TaskType

if TYPE_CHECKING:
    from collections.abc import Callable

    from getitune.metrics import MetricCallable
    from getitune.types.types import ANNOTATIONS, DATA, METRICS, MODEL

logger = logging.getLogger()


class OVEngine(Engine):
    """OV Engine.

    This class defines the OV Engine for getitune, which governs each step of the OpenVINO validation workflow.
    Supports both OpenVINO IR (.xml) and ONNX (.onnx) model formats for test and predict operations.
    """

    _SUPPORTED_MODEL_SUFFIXES: ClassVar[list[str]] = [".xml", ".onnx"]

    def __init__(
        self,
        data: DataModule | PathLike,
        model: OVModel | PathLike,
        work_dir: PathLike = "./getitune-workspace",
    ):
        """Initialize the OVEngine.

        Args:
            data (DataModule | PathLike): The data module or path to the data root.
                If a path is provided, the engine will automatically create a datamodule
                based on the data root and model.
            model (OVModel | PathLike): The OV model for the engine.
                A PathLike object to an OpenVINO IR XML or ONNX file can also be provided.
            work_dir (PathLike, optional): Working directory for the engine. Defaults to "./getitune-workspace".
        """
        self._work_dir = work_dir
        if isinstance(model, (str, os.PathLike)) and Path(model).suffix in self._SUPPORTED_MODEL_SUFFIXES:
            task: TaskType | None = self._derive_task_from_model(model)
        elif isinstance(model, OVModel):
            task = model.task  # type: ignore[assignment]
        else:
            msg = "Please provide a valid OpenVINO model or a path to an OpenVINO IR XML/.onnx file."
            raise ValueError(msg)
        self._auto_configurator = AutoConfigurator(
            data_root=data if isinstance(data, (str, os.PathLike)) else None,
            task=task,
        )

        if isinstance(data, DataModule):
            if task is not None and data.task != task:
                msg = (
                    "The task of the provided datamodule does not match the task derived from the model. "
                    f"datamodule.task={data.task}, model.task={task}"
                )
                raise ValueError(msg)
            self._datamodule: DataModule | None = data
        else:
            self._datamodule = self._auto_configurator.get_datamodule()

        self._model: OVModel = model if isinstance(model, OVModel) else self._auto_configurator.get_ov_model(model)

    def _derive_task_from_ir(self, ir_xml: PathLike) -> TaskType:
        """Derive the task type from the IR model XML file.

        Args:
            ir_xml (PathLike): Path to the IR model XML file.

        Returns:
            TaskType: The derived task type.

        Raises:
            ValueError: If the task type is unsupported or the XML file is invalid.
        """
        task_map = {
            "classification_mlc": TaskType.MULTI_LABEL_CLS,
            "classification_mc": TaskType.MULTI_CLASS_CLS,
            "segmentation": TaskType.SEMANTIC_SEGMENTATION,
            "detection": TaskType.DETECTION,
            "instance_segmentation": TaskType.INSTANCE_SEGMENTATION,
            "keypoint_detection": TaskType.KEYPOINT_DETECTION,
        }

        tree = Elet.parse(ir_xml)
        root = tree.getroot()
        rt_info = root.find("rt_info")
        if rt_info is None:
            msg = "No <rt_info> found in the IR model XML file. Please check the model file."
            raise ValueError(msg)

        task_type = rt_info.find(".//task_type")
        if task_type is None:
            msg = (
                "No <task_type> found in the IR model XML file. Please check the model file."
                "Task cannot be derived from the model."
            )
            raise ValueError(msg)
        task_type = task_type.attrib.get("value")

        if task_type == "classification":
            hierarchical = rt_info.find(".//hierarchical")
            if hierarchical is not None and hierarchical.attrib.get("value") == "True":
                msg = "Hierarchical classification models are no longer supported."
                raise ValueError(msg)
            multilabel = rt_info.find(".//multilabel")
            if multilabel is None:
                msg = "No <multilabel> found for the classification model in the IR metadata."
                raise ValueError(msg)
            task_name = task_type + ("_mlc" if multilabel.attrib.get("value") == "True" else "_mc")
        else:
            task_name = task_type

        if task_name not in task_map:
            msg = f"Unsupported task type '{task_name}' derived from the IR model XML file."
            raise ValueError(msg)

        return task_map[task_name]

    def _derive_task_from_onnx(self, onnx_path: PathLike) -> TaskType:
        """Derive the task type from ONNX model metadata_props.

        Args:
            onnx_path (PathLike): Path to the ONNX model file.

        Returns:
            TaskType: The derived task type.

        Raises:
            ValueError: If the task type is unsupported or the ONNX metadata is missing.
        """
        task_map = {
            "classification_mlc": TaskType.MULTI_LABEL_CLS,
            "classification_mc": TaskType.MULTI_CLASS_CLS,
            "segmentation": TaskType.SEMANTIC_SEGMENTATION,
            "detection": TaskType.DETECTION,
            "instance_segmentation": TaskType.INSTANCE_SEGMENTATION,
            "keypoint_detection": TaskType.KEYPOINT_DETECTION,
        }

        onnx_model = onnx.load(str(onnx_path), load_external_data=False)
        metadata = load_parameters_from_onnx(onnx_model)

        model_info = metadata.get("model_info", {})
        task_type = model_info.get("task_type")
        if task_type is None:
            msg = "No 'task_type' found in ONNX model metadata. Please ensure the model was exported by getitune."
            raise ValueError(msg)

        if task_type == "classification":
            if model_info.get("hierarchical") == "True":
                msg = "Hierarchical classification models are no longer supported."
                raise ValueError(msg)
            if "multilabel" not in model_info:
                msg = "No 'multilabel' found for the classification model in ONNX metadata."
                raise ValueError(msg)
            task_name = task_type + ("_mlc" if model_info.get("multilabel") == "True" else "_mc")
        else:
            task_name = task_type

        if task_name not in task_map:
            msg = f"Unsupported task type '{task_name}' derived from the ONNX model metadata."
            raise ValueError(msg)

        return task_map[task_name]

    def _derive_task_from_model(self, model_path: PathLike) -> TaskType:
        """Derive the task type from a model file (.xml or .onnx).

        Args:
            model_path (PathLike): Path to the model file.

        Returns:
            TaskType: The derived task type.

        Raises:
            ValueError: If the model format is unsupported.
        """
        path = Path(str(model_path))
        if path.suffix == ".xml":
            return self._derive_task_from_ir(model_path)
        if path.suffix == ".onnx":
            return self._derive_task_from_onnx(model_path)
        msg = f"Unsupported model format: '{path.suffix}'. Supported formats: {self._SUPPORTED_MODEL_SUFFIXES}"
        raise ValueError(msg)

    @property
    def _is_onnx(self) -> bool:
        """Check if the currently loaded model is an ONNX model."""
        return self.model is not None and Path(str(self.model.model_path)).suffix == ".onnx"

    def train(self, *args, **kwargs) -> METRICS:
        """Train method is not supported for OVEngine."""
        msg = "OVEngine does not support training. Use test or predict methods to evaluate IR model."
        raise NotImplementedError(msg)

    def export(self, *args, **kwargs) -> Path:
        """Export method is not supported for OVEngine."""
        msg = "OVEngine does not support export."
        raise NotImplementedError(msg)

    def test(
        self,
        data: DataModule | PathLike | None = None,
        checkpoint: PathLike | None = None,
        metric: MetricCallable | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
        **kwargs,
    ) -> METRICS:
        """Run the testing phase of the engine.

        Args:
            data (DataModule | PathLike | None, optional): The data to test on. It can be a data module
                or a path to the data root. If a path is provided, the engine will automatically
                create a datamodule based on the data root and model.
            checkpoint (PathLike | None, optional): Path to the checkpoint file to load the model from.
                Defaults to None.
            metric (MetricCallable | None, optional): If provided, overrides
                `LightningModel.metric_callable` with the given metric callable for evaluation.
            progress_callback (Callable[[int, int], None] | None, optional): Called after every
                test batch with ``(completed_batches, total_batches)``. OpenVINO evaluation runs
                on CPU and can take hours for large, high-resolution models; without a way to
                observe that it is progressing, an embedding application cannot distinguish a
                slow-but-healthy evaluation from a hung one. Exceptions raised by the callback
                propagate, so it can also be used to implement cooperative cancellation.

        Returns:
            METRICS: The computed metrics after testing the model on the provided data.
                (dictionary of metric names and values)

        Raises:
            RuntimeError: If required data or metric is not provided.
            ValueError: If label information between model and datamodule does not match.
        """
        if isinstance(data, (str, os.PathLike)):
            datamodule = self._auto_configurator.get_datamodule(data_root=data)
        elif isinstance(data, DataModule):
            datamodule = data
        else:
            datamodule = self.datamodule

        if datamodule is None:
            msg = "Please provide the `data` when creating the Engine, or pass it in OVEngine.test()."
            raise RuntimeError(msg)

        model = self._update_checkpoint(checkpoint)
        metric = metric or model.metric_callable

        # Propagate the datamodule tiling configuration to the model so that tile
        # predictions can be merged back to the original image during evaluation.
        if getattr(datamodule, "tile_config", None) is not None:
            model.tile_config = datamodule.tile_config

        datamodule = self._auto_configurator.update_ov_subset_pipeline(
            datamodule=datamodule,
            subset="test",
            task=model.task,
            input_size=model.input_size,
            keep_aspect_ratio=model.keep_aspect_ratio,
            center_padding=model.center_padding,
            pad_value=model.pad_value,
        )

        if metric is None:
            msg = "Please provide a `metric` when creating a OVModel or pass it in OVEngine.test()."
            raise RuntimeError(msg)

        if model.label_info != datamodule.label_info:
            msg = (
                "To launch a test pipeline, the label information should be same "
                "between the training and testing datasets. "
                "Please check whether you use the same dataset: "
                f"model.label_info={model.label_info}, "
                f"datamodule.label_info={self.datamodule.label_info}"
            )
            raise ValueError(msg)
        metric_callable = metric(datamodule.label_info)
        with Progress() as progress:
            dataloader = datamodule.test_dataloader()
            total_batches = len(dataloader)
            task = progress.add_task("Testing", total=total_batches)
            for completed, data_batch in enumerate(dataloader, start=1):
                model.test_step(data_batch, metric_callable)
                progress.update(task, advance=1)
                if progress_callback is not None:
                    progress_callback(completed, total_batches)

        metrics_result = model.compute_metrics(metric_callable)

        self.log_results(metrics_result)

        return metrics_result

    def log_results(self, metrics: METRICS) -> None:
        """Log testing phase results to a CSV file.

        This function behaves similarly to `LightningModel._log_metrics(metrics, key="test")`.
        """
        clean = {}
        for k, v in metrics.items():
            metric_name = f"test/{k}"
            if isinstance(v, torch.Tensor):
                if v.numel() == 1:
                    clean[metric_name] = v.item()
                else:
                    continue  # or flatten/log each value separately
            else:
                clean[metric_name] = v

        logger = CSVLogger(self.work_dir, name="csv/", prefix="")
        logger.log_metrics(clean, step=0)
        logger.finalize("success")

    def predict(
        self,
        data: DataModule | PathLike | list[np.array] | None = None,
        checkpoint: PathLike | None = None,
        explain: bool = False,
        explain_config: ExplainConfig | None = None,
        **kwargs,
    ) -> ANNOTATIONS:
        """Run predictions using the specified model and data.

        Args:
            data (DataModule | PathLike | list[np.array] | None, optional): The data module, path to data root,
                or a list of numpy images to use for predictions.
            checkpoint (PathLike | None, optional): The path to the IR XML file to load the model from.
            explain (bool, optional): Whether to generate "saliency_map" and "feature_vector". Defaults to False.
            explain_config (ExplainConfig | None, optional): Configuration for saliency map post-processing.

        Returns:
            ANNOTATIONS: The predictions made by the model on the provided data.
                (list of PredEntity)

        Raises:
            ValueError: If input data is invalid or label information does not match.
            TypeError: If input data type is unsupported.
        """
        from getitune.backend.lightning.models.utils.xai_utils import process_saliency_maps_in_pred_entity

        model = self._update_checkpoint(checkpoint)
        if isinstance(data, (str, os.PathLike)):
            data = self._auto_configurator.get_datamodule(data_root=data)

        datamodule = data or self.datamodule

        predict_result = []
        with Progress() as progress:
            if isinstance(datamodule, DataModule):
                if model.label_info != datamodule.label_info:
                    msg = (
                        "To launch a predict pipeline, the label information should be same "
                        "between the training and testing datasets. "
                        "Please check whether you use the same dataset: "
                        f"model.label_info={model.label_info}, "
                        f"datamodule.label_info={self.datamodule.label_info}"
                    )
                    raise ValueError(msg)

                # Propagate the datamodule tiling configuration to the model for predictions.
                if getattr(datamodule, "tile_config", None) is not None:
                    model.tile_config = datamodule.tile_config
                datamodule = self._auto_configurator.update_ov_subset_pipeline(
                    datamodule=datamodule,
                    subset="test",
                    task=model.task,
                    input_size=model.input_size,
                    keep_aspect_ratio=model.keep_aspect_ratio,
                    center_padding=model.center_padding,
                    pad_value=model.pad_value,
                )
                dataloader = datamodule.test_dataloader()
                task = progress.add_task("Predicting", total=len(dataloader))
                for data_batch in dataloader:
                    predict_result.append(model.predict_step(data_batch))
                    progress.update(task, advance=1)

            elif isinstance(datamodule, list):
                task = progress.add_task("Predicting", total=1)
                if len(datamodule) == 0:
                    msg = "The input data is empty."
                    raise ValueError(msg)
                if not isinstance(datamodule[0], np.ndarray):
                    msg = "The input data should be a list of numpy arrays."
                    raise TypeError(msg)
                customized_inputs = SampleBatch(
                    images=[torch.tensor(img) for img in datamodule],
                    imgs_info=[
                        ImageInfo(img_idx=i, ori_shape=img.shape, img_shape=img.shape)
                        for i, img in enumerate(datamodule)
                    ],
                )
                predict_result.append(model.predict_step(customized_inputs))
                progress.update(task, advance=1)
            else:
                msg = "The input data should be either a datamodule, valid path to data root or a list of numpy arrays."
                raise TypeError(msg)

        if explain and isinstance(datamodule, DataModule):
            if explain_config is None:
                explain_config = ExplainConfig()
            predict_result = process_saliency_maps_in_pred_entity(predict_result, explain_config)

        return predict_result

    def optimize(
        self,
        checkpoint: PathLike | None = None,
        datamodule: DataModule | None = None,
        max_data_subset_size: int | None = None,
        max_drop: float | None = None,
        max_num_iterations: int | None = None,
    ) -> Path:
        """Apply Post-Training Quantization (PTQ) to optimize the model.

        PTQ performs int-8 quantization on the input model, resulting in mixed precision.

        Note:
            Only OpenVINO IR (.xml) models are supported for optimization.
            ONNX models must be converted to OpenVINO IR format first.

        Args:
            checkpoint (PathLike | None, optional): Checkpoint to optimize. Defaults to None.
            datamodule (DataModule | None, optional): The data module to use for optimization.
            max_data_subset_size (int | None, optional): Maximum size of the train subset used for optimization.
                Defaults to None.
            max_drop (float | None, optional): Maximum accuracy drop allowed for accuracy-aware quantization.
                Defaults to None.
            max_num_iterations (int | None, optional): Maximum number of iterations for accuracy-aware
                quantization. ``None`` means unlimited. Only used when ``max_drop`` is set. Defaults to None.

        Returns:
            Path: Path to the optimized model.

        Raises:
            RuntimeError: If an ONNX model is used (not supported for optimization).
        """
        target_is_onnx = (checkpoint is not None and Path(str(checkpoint)).suffix == ".onnx") or (
            checkpoint is None and self._is_onnx
        )
        if target_is_onnx:
            msg = "OVEngine.optimize() does not support ONNX models. Please convert to OpenVINO IR format first."
            raise RuntimeError(msg)
        optimize_datamodule = datamodule if datamodule is not None else self.datamodule
        model = self._update_checkpoint(checkpoint)
        optimize_datamodule = self._auto_configurator.update_ov_subset_pipeline(
            datamodule=optimize_datamodule,
            subset="train",
            input_size=model.input_size,
            keep_aspect_ratio=model.keep_aspect_ratio,
            center_padding=model.center_padding,
            pad_value=model.pad_value,
        )

        ptq_config: dict[str, int | float] = {}
        if max_data_subset_size is not None:
            ptq_config["subset_size"] = max_data_subset_size
        if max_drop is not None:
            ptq_config["max_drop"] = max_drop
            if max_num_iterations is not None:
                ptq_config["max_num_iterations"] = max_num_iterations
        logger.debug(f"PTQ configuration: {ptq_config}")

        return model.optimize(
            Path(self.work_dir),
            optimize_datamodule,
            ptq_config,
        )

    @staticmethod
    def is_supported(model: MODEL, data: DATA) -> bool:
        """Check if the engine is supported for the given model and data."""
        check_model = False
        check_data = False
        if isinstance(model, OVModel):
            check_model = True
        elif isinstance(model, (str, os.PathLike)):
            model_path = Path(model)
            check_model = model_path.suffix in OVEngine._SUPPORTED_MODEL_SUFFIXES
        if isinstance(data, DataModule):
            check_data = True
        elif isinstance(data, (str, os.PathLike)):
            data_path = Path(data)
            check_data = data_path.exists()

        return check_model and check_data

    @classmethod
    def from_config(
        cls,
        config_path: PathLike,
        data: DataModule | PathLike | None = None,
        work_dir: PathLike | None = None,
        device: str | None = None,
        checkpoint: str | None = None,
        task: str | None = None,
        **kwargs,
    ) -> OVEngine:
        """OVEngine does not support construction from a recipe config.

        OpenVINO models are selected by passing a ``.xml`` or ``.onnx``
        weights path directly to :func:`~getitune.engine.create_engine` or
        as the *model* argument to :class:`OVEngine`.

        Args:
            config_path: Unused — included for API compatibility.
            data: Unused — included for API compatibility.
            work_dir: Unused — included for API compatibility.
            device: Unused — included for API compatibility.
            checkpoint: Unused — included for API compatibility.
            task: Unused — included for API compatibility.
            **kwargs: Unused — included for API compatibility.

        Raises:
            NotImplementedError: Always raised.
        """
        msg = (
            f"OVEngine does not support construction from a recipe config '{config_path}'. "
            "Pass a .xml or .onnx model path directly to create_engine() instead."
        )
        raise NotImplementedError(msg)

    def _update_checkpoint(self, checkpoint: PathLike | None) -> OVModel:
        """Update the OVModel with the given checkpoint path.

        Args:
            checkpoint (PathLike | None): The new model file path (.xml or .onnx).

        Returns:
            OVModel: The updated OVModel instance.

        Raises:
            ValueError: If no model or checkpoint is provided.
            RuntimeError: If the checkpoint file format is unsupported.
        """
        if checkpoint is None and self.model is None:
            msg = "Please provide either a model or a checkpoint path."
            raise ValueError(msg)
        if checkpoint is not None and Path(str(checkpoint)).suffix not in self._SUPPORTED_MODEL_SUFFIXES:
            msg = (
                f"OVEngine supports only {self._SUPPORTED_MODEL_SUFFIXES} checkpoints, "
                f"got '{Path(str(checkpoint)).suffix}'"
            )
            raise RuntimeError(msg)
        if checkpoint is not None:
            task = self._derive_task_from_model(checkpoint)
            return self._auto_configurator.get_ov_model(model_name=str(checkpoint), task=task)

        return self.model  # type: ignore[return-value]

    @property
    def work_dir(self) -> PathLike:
        """Get the working directory.

        Returns:
            PathLike: The working directory path.
        """
        return self._work_dir

    @work_dir.setter
    def work_dir(self, work_dir: PathLike) -> None:
        """Set the working directory.

        Args:
            work_dir (PathLike): The new working directory path.
        """
        self._work_dir = work_dir

    @property
    def model(self) -> OVModel:
        """Get the model associated with the engine.

        Returns:
            OVModel: The OVModel object or None if not set.
        """
        return self._model

    @property
    def datamodule(self) -> DataModule:
        """Get the datamodule associated with the engine.

        Returns:
            DataModule: The DataModule object.

        Raises:
            RuntimeError: If the datamodule is not set.
        """
        if self._datamodule is None:
            msg = "Please include the `data_root` or `datamodule` when creating the Engine."
            raise RuntimeError(msg)
        return self._datamodule

    @property
    def best_checkpoint(self) -> Path | None:
        """OVEngine does not produce checkpoints."""
        return None
