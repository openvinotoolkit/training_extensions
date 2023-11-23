"""Task of OTX Classification."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import io
import json
import os
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
import torch

from otx.algorithms.classification.configs.base import ClassificationConfig
from otx.algorithms.classification.utils import (
    get_cls_deploy_config,
    get_cls_inferencer_configuration,
    get_cls_model_api_configuration,
    get_hierarchical_label_list,
)
from otx.algorithms.classification.utils import (
    get_multihead_class_info as get_hierarchical_info,
)
from otx.algorithms.common.configs import TrainType
from otx.algorithms.common.configs.configuration_enums import InputSizePreset
from otx.algorithms.common.tasks.base_task import TRAIN_TYPE_DIR_PATH, OTXTask
from otx.algorithms.common.utils import embed_ir_model_data
from otx.algorithms.common.utils.callback import TrainingProgressCallback
from otx.algorithms.common.utils.utils import embed_onnx_model_data
from otx.api.configuration import cfg_helper
from otx.api.configuration.helper.utils import ids_to_strings
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.explain_parameters import ExplainParameters
from otx.api.entities.inference_parameters import (
    InferenceParameters,
)
from otx.api.entities.inference_parameters import (
    default_progress_callback as default_infer_progress_callback,
)
from otx.api.entities.label import LabelEntity
from otx.api.entities.label_schema import LabelGroup
from otx.api.entities.metadata import FloatMetadata, FloatType
from otx.api.entities.metrics import (
    CurveMetric,
    LineChartInfo,
    LineMetricsGroup,
    MetricsGroup,
    Performance,
    ScoreMetric,
)
from otx.api.entities.model import (
    ModelEntity,
    ModelPrecision,
)
from otx.api.entities.resultset import ResultSetEntity
from otx.api.entities.scored_label import ScoredLabel
from otx.api.entities.task_environment import TaskEnvironment
from otx.api.entities.tensor import TensorEntity
from otx.api.entities.train_parameters import (
    TrainParameters,
)
from otx.api.entities.train_parameters import (
    default_progress_callback as default_train_progress_callback,
)
from otx.api.serialization.label_mapper import label_schema_to_bytes
from otx.api.usecases.evaluation.metrics_helper import MetricsHelper
from otx.api.usecases.tasks.interfaces.export_interface import ExportType
from otx.api.utils.dataset_utils import add_saliency_maps_to_dataset_item
from otx.api.utils.labels_utils import get_empty_label
from otx.cli.utils.multi_gpu import is_multigpu_child_process
from otx.core.data.caching.mem_cache_handler import MemCacheHandlerSingleton
from otx.utils.logger import get_logger

logger = get_logger()
RECIPE_TRAIN_TYPE = {
    TrainType.Semisupervised: "semisl.yaml",
    TrainType.Incremental: "incremental.yaml",
    TrainType.Selfsupervised: "selfsl.yaml",
}


class OTXClassificationTask(OTXTask, ABC):
    """Task class for OTX classification."""

    # pylint: disable=too-many-instance-attributes, too-many-locals, too-many-boolean-expressions
    def __init__(self, task_environment: TaskEnvironment, output_path: Optional[str] = None):
        super().__init__(task_environment, output_path)
        self._task_config = ClassificationConfig
        self._hyperparams = self._task_environment.get_hyper_parameters(self._task_config)
        if len(self._task_environment.get_labels(False)) == 1:
            self._labels = self._task_environment.get_labels(include_empty=True)
        else:
            self._labels = self._task_environment.get_labels(include_empty=False)
        self._empty_label = get_empty_label(self._task_environment.label_schema)

        self._multilabel = False
        self._hierarchical = False
        self._hierarchical_info = None
        self._selfsl = False
        self._set_train_mode()

        self._train_type = self._hyperparams.algo_backend.train_type
        self._model_dir = os.path.join(
            os.path.abspath(os.path.dirname(self._task_environment.model_template.model_template_path)),
            TRAIN_TYPE_DIR_PATH[self._train_type.name],
        )
        if (
            self._train_type in RECIPE_TRAIN_TYPE
            and self._train_type == TrainType.Incremental
            and not self._multilabel
            and not self._hierarchical
            and self._hyperparams.learning_parameters.enable_supcon
            and not self._model_dir.endswith("supcon")
        ):
            self._model_dir = os.path.join(self._model_dir, "supcon")

        self.data_pipeline_path = os.path.join(self._model_dir, "data_pipeline.py")

        if self._task_environment.model is not None:
            self._load_model()
        if hasattr(self._hyperparams.learning_parameters, "input_size"):
            input_size_cfg = InputSizePreset(self._hyperparams.learning_parameters.input_size.value)
        else:
            input_size_cfg = InputSizePreset.DEFAULT
        self._input_size = input_size_cfg.tuple

        if hasattr(self._hyperparams.learning_parameters, "input_size"):
            input_size_cfg = InputSizePreset(self._hyperparams.learning_parameters.input_size.value)
        else:
            input_size_cfg = InputSizePreset.DEFAULT
        self._input_size = input_size_cfg.tuple

    def _is_multi_label(self, label_groups: List[LabelGroup], all_labels: List[LabelEntity]):
        """Check whether the current training mode is multi-label or not."""
        # NOTE: In the current Geti, multi-label should have `___` symbol for all group names.
        find_multilabel_symbol = ["___" in getattr(i, "name", "") for i in label_groups]
        return (
            (len(label_groups) > 1) and (len(label_groups) == len(all_labels)) and (False not in find_multilabel_symbol)
        )

    def _set_train_mode(self):
        label_groups = self._task_environment.label_schema.get_groups(include_empty=False)
        all_labels = self._task_environment.label_schema.get_labels(include_empty=False)

        self._multilabel = self._is_multi_label(label_groups, all_labels)
        if self._multilabel:
            logger.info("Classification mode: multilabel")
        elif len(label_groups) > 1:
            logger.info("Classification mode: hierarchical")
            self._hierarchical = True
            self._hierarchical_info = get_hierarchical_info(self._task_environment.label_schema)
        if not self._multilabel and not self._hierarchical:
            logger.info("Classification mode: multiclass")

        if self._hyperparams.algo_backend.train_type == TrainType.Selfsupervised:
            self._selfsl = True

    def infer(
        self,
        dataset: DatasetEntity,
        inference_parameters: Optional[InferenceParameters] = None,
    ) -> DatasetEntity:
        """Main infer function of OTX Classification."""

        logger.info("infer()")

        results = self._infer_model(dataset, inference_parameters)
        prediction_results = zip(
            results["eval_predictions"],
            results["feature_vectors"],
            results["saliency_maps"],
        )

        update_progress_callback = default_infer_progress_callback
        process_saliency_maps = False
        explain_predicted_classes = True
        if inference_parameters is not None:
            update_progress_callback = inference_parameters.update_progress  # type: ignore
            process_saliency_maps = inference_parameters.process_saliency_maps
            explain_predicted_classes = inference_parameters.explain_predicted_classes

        self._add_predictions_to_dataset(
            prediction_results, dataset, update_progress_callback, process_saliency_maps, explain_predicted_classes
        )
        return dataset

    def train(
        self,
        dataset: DatasetEntity,
        output_model: ModelEntity,
        train_parameters: Optional[TrainParameters] = None,
        seed: Optional[int] = None,
        deterministic: bool = False,
    ):
        """Train function for OTX classification task.

        Actual training is processed by _train_model fucntion
        """
        logger.info("train()")
        # Check for stop signal when training has stopped.
        # If should_stop is true, training was cancelled and no new
        if self._should_stop:
            logger.info("Training cancelled.")
            self._should_stop = False
            self._is_training = False
            return
        self.seed = seed
        self.deterministic = deterministic

        # Set OTX LoggerHook & Time Monitor
        if train_parameters:
            update_progress_callback = train_parameters.update_progress
        else:
            update_progress_callback = default_train_progress_callback
        self._time_monitor = TrainingProgressCallback(update_progress_callback)

        results = self._train_model(dataset)

        MemCacheHandlerSingleton.delete()

        # Check for stop signal when training has stopped. If should_stop is true, training was cancelled and no new
        if self._should_stop:
            logger.info("Training cancelled.")
            self._should_stop = False
            self._is_training = False
            return

        # get output model
        model_ckpt = results.get("final_ckpt")
        if model_ckpt is None:
            logger.error("cannot find final checkpoint from the results.")
            return
        # update checkpoint to the newly trained model
        self._model_ckpt = model_ckpt

        # compose performance statistics
        training_metrics, final_acc = self._generate_training_metrics(self._learning_curves)
        # save resulting model
        self.save_model(output_model)
        performance = Performance(
            score=ScoreMetric(value=final_acc, name="accuracy"),
            dashboard_metrics=training_metrics,
        )
        logger.info(f"Final model performance: {str(performance)}")
        output_model.performance = performance
        self._is_training = False
        logger.info("train done.")

    def export(
        self,
        export_type: ExportType,
        output_model: ModelEntity,
        precision: ModelPrecision = ModelPrecision.FP32,
        dump_features: bool = True,
    ):
        """Export function of OTX Classification Task."""

        logger.info("Exporting the model")

        self._update_model_export_metadata(output_model, export_type, precision, dump_features)
        results = self._export_model(precision, export_type, dump_features)
        outputs = results.get("outputs")
        logger.debug(f"results of run_task = {outputs}")
        if outputs is None:
            raise RuntimeError(results.get("msg"))

        inference_config = get_cls_inferencer_configuration(self._task_environment.label_schema)
        extra_model_data = get_cls_model_api_configuration(self._task_environment.label_schema, inference_config)
        if export_type == ExportType.ONNX:
            extra_model_data[("model_info", "mean_values")] = results.get("inference_parameters").get("mean_values")
            extra_model_data[("model_info", "scale_values")] = results.get("inference_parameters").get("scale_values")

            onnx_file = outputs.get("onnx")
            embed_onnx_model_data(onnx_file, extra_model_data)
            with open(onnx_file, "rb") as f:
                output_model.set_data("model.onnx", f.read())
        else:
            bin_file = outputs.get("bin")
            xml_file = outputs.get("xml")

            deploy_cfg = get_cls_deploy_config(self._task_environment.label_schema, inference_config)
            extra_model_data[("otx_config",)] = json.dumps(deploy_cfg, ensure_ascii=False)
            embed_ir_model_data(xml_file, extra_model_data)

            with open(bin_file, "rb") as f:
                output_model.set_data("openvino.bin", f.read())
            with open(xml_file, "rb") as f:
                output_model.set_data("openvino.xml", f.read())

        output_model.set_data(
            "label_schema.json",
            label_schema_to_bytes(self._task_environment.label_schema),
        )
        logger.info("Exporting completed")

    def explain(
        self,
        dataset: DatasetEntity,
        explain_parameters: Optional[ExplainParameters] = None,
    ) -> DatasetEntity:
        """Main explain function of OTX Classification Task."""

        predictions, saliency_maps = self._explain_model(
            dataset,
            explain_parameters=explain_parameters,
        )

        update_progress_callback = default_infer_progress_callback
        process_saliency_maps = False
        explain_predicted_classes = True
        if explain_parameters is not None:
            update_progress_callback = explain_parameters.update_progress  # type: ignore
            process_saliency_maps = explain_parameters.process_saliency_maps
            explain_predicted_classes = explain_parameters.explain_predicted_classes

        self._add_explanations_to_dataset(
            predictions,
            saliency_maps,
            dataset,
            update_progress_callback,
            process_saliency_maps,
            explain_predicted_classes,
        )
        logger.info("Explain completed")
        return dataset

    def evaluate(
        self,
        output_resultset: ResultSetEntity,
        evaluation_metric: Optional[str] = None,
    ):
        """Evaluate function of OTX Classification Task."""

        logger.info("called evaluate()")
        if evaluation_metric is not None:
            logger.warning(
                f"Requested to use {evaluation_metric} metric, " "but parameter is ignored. Use accuracy instead."
            )
        metric = MetricsHelper.compute_accuracy(output_resultset)
        logger.info(f"Accuracy after evaluation: {metric.accuracy.value}")
        output_resultset.performance = metric.get_performance()
        logger.info("Evaluation completed")

    # pylint: disable=too-many-branches, too-many-locals
    def _add_predictions_to_dataset(
        self,
        prediction_results,
        dataset,
        update_progress_callback,
        process_saliency_maps=False,
        explain_predicted_classes=True,
    ):
        """Loop over dataset again to assign predictions.Convert from MMClassification format to OTX format."""

        dataset_size = len(dataset)
        pos_thr = 0.5
        label_list = self._labels
        # Fix the order for hierarchical labels to adjust classes with model outputs
        if self._hierarchical:
            label_list = get_hierarchical_label_list(self._hierarchical_info, label_list)
        for i, (dataset_item, prediction_items) in enumerate(zip(dataset, prediction_results)):
            prediction_item, feature_vector, saliency_map = prediction_items
            if any(np.isnan(prediction_item)):
                logger.info("Nan in prediction_item.")

            item_labels = self._get_item_labels(prediction_item, pos_thr)

            dataset_item.append_labels(item_labels)

            probs = TensorEntity(name="probabilities", numpy=prediction_item.reshape(-1))
            dataset_item.append_metadata_item(probs, model=self._task_environment.model)

            top_idxs = np.argpartition(prediction_item, -2)[-2:]
            top_probs = prediction_item[top_idxs]
            active_score_media = FloatMetadata(
                name="active_score", value=top_probs[1] - top_probs[0], float_type=FloatType.ACTIVE_SCORE
            )
            dataset_item.append_metadata_item(active_score_media, model=self._task_environment.model)

            if feature_vector is not None:
                active_score = TensorEntity(name="representation_vector", numpy=feature_vector.reshape(-1))
                dataset_item.append_metadata_item(active_score, model=self._task_environment.model)

            if saliency_map is not None:
                add_saliency_maps_to_dataset_item(
                    dataset_item=dataset_item,
                    saliency_map=saliency_map,
                    model=self._task_environment.model,
                    labels=label_list,
                    predicted_scored_labels=item_labels,
                    explain_predicted_classes=explain_predicted_classes,
                    process_saliency_maps=process_saliency_maps,
                )
            update_progress_callback(int(i / dataset_size * 100))

    # pylint: disable=too-many-locals
    def _get_item_labels(self, prediction_item, pos_thr):
        item_labels = []
        if self._multilabel:
            if max(prediction_item) < pos_thr:
                logger.info("Confidence is smaller than pos_thr, empty_label will be appended to item_labels.")
                item_labels.append(ScoredLabel(self._empty_label, probability=1.0))
            else:
                for cls_idx, pred_item in enumerate(prediction_item):
                    if pred_item > pos_thr:
                        cls_label = ScoredLabel(self._labels[cls_idx], probability=float(pred_item))
                        item_labels.append(cls_label)

        elif self._hierarchical:
            for head_idx in range(self._hierarchical_info["num_multiclass_heads"]):
                logits_begin, logits_end = self._hierarchical_info["head_idx_to_logits_range"][str(head_idx)]
                head_logits = prediction_item[logits_begin:logits_end]
                head_pred = np.argmax(head_logits)  # Assume logits already passed softmax
                label_str = self._hierarchical_info["all_groups"][head_idx][head_pred]
                otx_label = next(x for x in self._labels if x.name == label_str)
                item_labels.append(ScoredLabel(label=otx_label, probability=float(head_logits[head_pred])))

            if self._hierarchical_info["num_multilabel_classes"]:
                head_logits = prediction_item[self._hierarchical_info["num_single_label_classes"] :]
                for logit_idx, logit in enumerate(head_logits):
                    if logit > pos_thr:  # Assume logits already passed sigmoid
                        label_str_idx = self._hierarchical_info["num_multiclass_heads"] + logit_idx
                        label_str = self._hierarchical_info["all_groups"][label_str_idx][0]
                        otx_label = next(x for x in self._labels if x.name == label_str)
                        item_labels.append(ScoredLabel(label=otx_label, probability=float(logit)))
            item_labels = self._task_environment.label_schema.resolve_labels_greedily(item_labels)
            if not item_labels:
                logger.info("item_labels is empty.")
                item_labels.append(ScoredLabel(self._empty_label, probability=1.0))

        else:
            label_idx = prediction_item.argmax()
            cls_label = ScoredLabel(
                self._labels[label_idx],
                probability=float(prediction_item[label_idx]),
            )
            item_labels.append(cls_label)
        return item_labels

    def _add_explanations_to_dataset(
        self,
        predictions,
        saliency_maps,
        dataset,
        update_progress_callback,
        process_saliency_maps,
        explain_predicted_classes,
    ):
        """Loop over dataset again and assign saliency maps."""
        dataset_size = len(dataset)
        label_list = self._labels
        # Fix the order for hierarchical labels to adjust classes with model outputs
        if self._hierarchical:
            label_list = get_hierarchical_label_list(self._hierarchical_info, label_list)
        for i, (dataset_item, prediction_item, saliency_map) in enumerate(zip(dataset, predictions, saliency_maps)):
            item_labels = self._get_item_labels(prediction_item, pos_thr=0.5)
            add_saliency_maps_to_dataset_item(
                dataset_item=dataset_item,
                saliency_map=saliency_map,
                model=self._task_environment.model,
                labels=label_list,
                predicted_scored_labels=item_labels,
                explain_predicted_classes=explain_predicted_classes,
                process_saliency_maps=process_saliency_maps,
            )
            update_progress_callback(int(i / dataset_size * 100))

    def save_model(self, output_model: ModelEntity):
        """Save best model weights in ClassificationTrainTask."""
        if is_multigpu_child_process():
            return

        logger.info("called save_model")
        buffer = io.BytesIO()
        hyperparams_str = ids_to_strings(cfg_helper.convert(self._hyperparams, dict, enum_to_str=True))
        labels = {label.name: label.color.rgb_tuple for label in self._labels}
        model_ckpt = torch.load(self._model_ckpt)
        modelinfo = {
            "model": model_ckpt,
            "config": hyperparams_str,
            "labels": labels,
            "input_size": self._input_size,
            "VERSION": 1,
        }

        torch.save(modelinfo, buffer)
        output_model.set_data("weights.pth", buffer.getvalue())
        output_model.set_data(
            "label_schema.json",
            label_schema_to_bytes(self._task_environment.label_schema),
        )
        output_model.precision = self._precision

    def _generate_training_metrics(self, learning_curves):  # pylint: disable=arguments-renamed
        """Parses the classification logs to get metrics from the latest training run.

        :return output List[MetricsGroup]
        """
        output: List[MetricsGroup] = []

        if self._multilabel:
            metric_key = "val/accuracy-mlc"
        elif self._hierarchical:
            metric_key = "val/MHAcc"
        else:
            metric_key = "val/accuracy (%)"

        # Learning curves
        best_acc = -1
        if learning_curves is None:
            return output

        for key, curve in learning_curves.items():
            metric_curve = CurveMetric(xs=curve.x, ys=curve.y, name=key)
            if key == metric_key:
                best_acc = max(curve.y)
            visualization_info = LineChartInfo(name=key, x_axis_label="Timestamp", y_axis_label=key)
            output.append(LineMetricsGroup(metrics=[metric_curve], visualization_info=visualization_info))

        return output, best_acc

    @abstractmethod
    def _train_model(self, dataset: DatasetEntity):
        """Train model and return the results."""
        raise NotImplementedError

    @abstractmethod
    def _export_model(self, precision: ModelPrecision, export_format: ExportType, dump_features: bool):
        """Export model and return the results."""
        raise NotImplementedError

    @abstractmethod
    def _explain_model(self, dataset: DatasetEntity, explain_parameters: Optional[ExplainParameters]):
        """Explain model and return the results."""
        raise NotImplementedError

    @abstractmethod
    def _infer_model(
        self,
        dataset: DatasetEntity,
        inference_parameters: Optional[InferenceParameters] = None,
    ):
        """Get inference results from dataset."""
        raise NotImplementedError
