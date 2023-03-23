"""Inference Task of OTX Classification."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import json
import os
from typing import Optional

import numpy as np
from mmcv.utils import ConfigDict

from otx.algorithms.classification.adapters.mmcls.utils.builder import build_classifier
from otx.algorithms.classification.adapters.mmcls.utils.config_utils import (
    patch_datasets,
    patch_evaluation,
)
from otx.algorithms.classification.configs import ClassificationConfig
from otx.algorithms.classification.utils import (
    get_cls_deploy_config,
    get_cls_inferencer_configuration,
    get_cls_model_api_configuration,
)
from otx.algorithms.classification.utils import (
    get_multihead_class_info as get_hierarchical_info,
)
from otx.algorithms.common.adapters.mmcv.utils import (
    patch_data_pipeline,
    patch_default_config,
    patch_runner,
)
from otx.algorithms.common.adapters.mmcv.utils.config_utils import MPAConfig
from otx.algorithms.common.configs import TrainType
from otx.algorithms.common.tasks import BaseTask
from otx.algorithms.common.utils import embed_ir_model_data
from otx.algorithms.common.utils.logger import get_logger
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.inference_parameters import (
    InferenceParameters,
    default_progress_callback,
)
from otx.api.entities.metadata import FloatMetadata, FloatType
from otx.api.entities.model import (  # ModelStatus
    ModelEntity,
    ModelFormat,
    ModelOptimizationType,
    ModelPrecision,
)
from otx.api.entities.resultset import ResultSetEntity
from otx.api.entities.scored_label import ScoredLabel
from otx.api.entities.task_environment import TaskEnvironment
from otx.api.entities.tensor import TensorEntity
from otx.api.serialization.label_mapper import label_schema_to_bytes
from otx.api.usecases.evaluation.metrics_helper import MetricsHelper
from otx.api.usecases.tasks.interfaces.evaluate_interface import IEvaluationTask
from otx.api.usecases.tasks.interfaces.explain_interface import IExplainTask
from otx.api.usecases.tasks.interfaces.export_interface import ExportType, IExportTask
from otx.api.usecases.tasks.interfaces.inference_interface import IInferenceTask
from otx.api.usecases.tasks.interfaces.unload_interface import IUnload
from otx.api.utils.argument_checks import (
    DatasetParamTypeCheck,
    check_input_parameters_type,
)
from otx.api.utils.dataset_utils import add_saliency_maps_to_dataset_item
from otx.api.utils.labels_utils import get_empty_label

# pylint: disable=invalid-name

logger = get_logger()

TASK_CONFIG = ClassificationConfig
RECIPE_TRAIN_TYPE = {
    TrainType.Semisupervised: "semisl.yaml",
    TrainType.Incremental: "incremental.yaml",
    TrainType.Selfsupervised: "selfsl.yaml",
}


# pylint: disable=too-many-instance-attributes
class ClassificationInferenceTask(BaseTask, IInferenceTask, IExportTask, IEvaluationTask, IExplainTask, IUnload):
    """Inference Task Implementation of OTX Classification."""

    @check_input_parameters_type()
    def __init__(self, task_environment: TaskEnvironment, **kwargs):
        self._should_stop = False
        super().__init__(TASK_CONFIG, task_environment, **kwargs)

        self._task_environment = task_environment
        if len(task_environment.get_labels(False)) == 1:
            self._labels = task_environment.get_labels(include_empty=True)
        else:
            self._labels = task_environment.get_labels(include_empty=False)
        self._empty_label = get_empty_label(task_environment.label_schema)

        self._multilabel = False
        self._hierarchical = False
        self._selfsl = False

        self._multilabel = len(task_environment.label_schema.get_groups(False)) > 1 and len(
            task_environment.label_schema.get_groups(False)
        ) == len(
            task_environment.get_labels(include_empty=False)
        )  # noqa:E127
        if self._multilabel:
            logger.info("Classification mode: multilabel")

        self._hierarchical_info = None
        if not self._multilabel and len(task_environment.label_schema.get_groups(False)) > 1:
            logger.info("Classification mode: hierarchical")
            self._hierarchical = True
            self._hierarchical_info = get_hierarchical_info(task_environment.label_schema)
        if not self._multilabel and not self._hierarchical:
            logger.info("Classification mode: multiclass")

        if self._hyperparams.algo_backend.train_type == TrainType.Selfsupervised:
            self._selfsl = True

    @check_input_parameters_type({"dataset": DatasetParamTypeCheck})
    def infer(
        self,
        dataset: DatasetEntity,
        inference_parameters: Optional[InferenceParameters] = None,
    ) -> DatasetEntity:
        """Main infer function of OTX Classification."""

        logger.info("called infer()")
        stage_module = "ClsInferrer"
        self._data_cfg = self._init_test_data_cfg(dataset)
        dataset = dataset.with_empty_annotations()

        dump_features = True
        dump_saliency_map = not inference_parameters.is_evaluation if inference_parameters else True

        results = self._run_task(
            stage_module,
            mode="eval",
            dataset=dataset,
            dump_features=dump_features,
            dump_saliency_map=dump_saliency_map,
        )
        logger.debug(f"result of run_task {stage_module} module = {results}")
        predictions = results["outputs"]
        prediction_results = zip(
            predictions["eval_predictions"],
            predictions["feature_vectors"],
            predictions["saliency_maps"],
        )

        update_progress_callback = default_progress_callback
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

    def explain(
        self,
        dataset: DatasetEntity,
        explain_parameters: Optional[InferenceParameters] = None,
    ) -> DatasetEntity:
        """Main explain function of OTX Classification Task."""
        logger.info("called explain()")
        stage_module = "ClsExplainer"
        self._data_cfg = self._init_test_data_cfg(dataset)
        dataset = dataset.with_empty_annotations()

        results = self._run_task(
            stage_module,
            mode="train",
            dataset=dataset,
            explainer=explain_parameters.explainer if explain_parameters else None,
        )
        logger.debug(f"result of run_task {stage_module} module = {results}")
        predictions = results["outputs"]["eval_predictions"]
        saliency_maps = results["outputs"]["saliency_maps"]

        update_progress_callback = default_progress_callback
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

    @check_input_parameters_type()
    def evaluate(
        self,
        output_resultset: ResultSetEntity,
        evaluation_metric: Optional[str] = None,
    ):
        """Evaluate function of OTX Classification Task."""

        logger.info("called evaluate()")
        metric = MetricsHelper.compute_accuracy(output_resultset)
        logger.info(f"Accuracy after evaluation: {metric.accuracy.value}")
        output_resultset.performance = metric.get_performance()
        logger.info("Evaluation completed")

    def unload(self):
        """Unload function of OTX Classification Task."""
        logger.info("called unload()")
        self.cleanup()

    @check_input_parameters_type()
    def export(
        self,
        export_type: ExportType,
        output_model: ModelEntity,
        precision: ModelPrecision = ModelPrecision.FP32,
        dump_features: bool = False,
    ):
        """Export function of OTX Classification Task."""

        logger.info("Exporting the model")
        if export_type != ExportType.OPENVINO:
            raise RuntimeError(f"not supported export type {export_type}")
        output_model.model_format = ModelFormat.OPENVINO
        output_model.optimization_type = ModelOptimizationType.MO

        stage_module = "ClsExporter"
        results = self._run_task(
            stage_module,
            mode="train",
            export=True,
            enable_fp16=(precision == ModelPrecision.FP16),
            dump_features=dump_features,
        )
        outputs = results.get("outputs")
        logger.debug(f"results of run_task = {outputs}")
        if outputs is None:
            raise RuntimeError(results.get("msg"))

        bin_file = outputs.get("bin")
        xml_file = outputs.get("xml")

        inference_config = get_cls_inferencer_configuration(self._task_environment.label_schema)
        deploy_cfg = get_cls_deploy_config(self._task_environment.label_schema, inference_config)
        ir_extra_data = get_cls_model_api_configuration(self._task_environment.label_schema, inference_config)
        ir_extra_data[("otx_config",)] = json.dumps(deploy_cfg, ensure_ascii=False)
        embed_ir_model_data(xml_file, ir_extra_data)

        if xml_file is None or bin_file is None:
            raise RuntimeError("invalid status of exporting. bin and xml should not be None")
        with open(bin_file, "rb") as f:
            output_model.set_data("openvino.bin", f.read())
        with open(xml_file, "rb") as f:
            output_model.set_data("openvino.xml", f.read())
        output_model.precision = self._precision
        output_model.set_data(
            "label_schema.json",
            label_schema_to_bytes(self._task_environment.label_schema),
        )
        logger.info("Exporting completed")

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
                        cls_label = ScoredLabel(self.labels[cls_idx], probability=float(pred_item))
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
            item_labels = self._task_environment.label_schema.resolve_labels_probabilistic(item_labels)
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
                    labels=self._labels,
                    predicted_scored_labels=item_labels,
                    explain_predicted_classes=explain_predicted_classes,
                    process_saliency_maps=process_saliency_maps,
                )
            update_progress_callback(int(i / dataset_size * 100))

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
        for i, (dataset_item, prediction_item, saliency_map) in enumerate(zip(dataset, predictions, saliency_maps)):
            item_labels = self._get_item_labels(prediction_item, pos_thr=0.5)
            add_saliency_maps_to_dataset_item(
                dataset_item=dataset_item,
                saliency_map=saliency_map,
                model=self._task_environment.model,
                labels=self._labels,
                predicted_scored_labels=item_labels,
                explain_predicted_classes=explain_predicted_classes,
                process_saliency_maps=process_saliency_maps,
            )
            update_progress_callback(int(i / dataset_size * 100))

    def _init_recipe_hparam(self) -> dict:
        params = self._hyperparams.learning_parameters
        warmup_iters = int(params.learning_rate_warmup_iters)
        if self._multilabel:
            # hack to use 1cycle policy
            lr_config = ConfigDict(max_lr=params.learning_rate, warmup=None)
        else:
            lr_config = (
                ConfigDict(warmup_iters=warmup_iters) if warmup_iters > 0 else ConfigDict(warmup_iters=0, warmup=None)
            )

        early_stop = False
        if self._recipe_cfg is not None:
            if params.enable_early_stopping and self._recipe_cfg.get("evaluation", None):
                early_stop = ConfigDict(
                    start=int(params.early_stop_start),
                    patience=int(params.early_stop_patience),
                    iteration_patience=int(params.early_stop_iteration_patience),
                )

        if self._recipe_cfg.runner.get("type").startswith("IterBasedRunner"):  # type: ignore
            runner = ConfigDict(max_iters=int(params.num_iters))
        else:
            runner = ConfigDict(max_epochs=int(params.num_iters))

        config = ConfigDict(
            optimizer=ConfigDict(lr=params.learning_rate),
            lr_config=lr_config,
            early_stop=early_stop,
            data=ConfigDict(
                samples_per_gpu=int(params.batch_size),
                workers_per_gpu=int(params.num_workers),
            ),
            runner=runner,
        )

        if self._train_type.value == "Semisupervised":
            unlabeled_config = ConfigDict(
                data=ConfigDict(
                    unlabeled_dataloader=ConfigDict(
                        samples_per_gpu=int(params.unlabeled_batch_size),
                        workers_per_gpu=int(params.num_workers),
                    )
                )
            )
            config.update(unlabeled_config)
        return config

    def _init_recipe(self):
        logger.info("called _init_recipe()")

        logger.info(f"train type = {self._train_type}")
        # TODO: Need to remove the hard coding for supcon only.
        # pylint: disable=too-many-boolean-expressions
        if (
            self._train_type in RECIPE_TRAIN_TYPE
            and self._train_type == TrainType.Incremental
            and not self._multilabel
            and not self._hierarchical
            and self._hyperparams.learning_parameters.enable_supcon
            and not self._model_dir.endswith("supcon")
        ):
            self._model_dir = os.path.join(self._model_dir, "supcon")

        self._recipe_cfg = self._init_model_cfg()

        # FIXME[Soobee] : if train type is not in cfg, it raises an error in default Incremental mode.
        # During semi-implementation, this line should be fixed to -> self._recipe_cfg.train_type = train_type
        self._recipe_cfg.train_type = self._train_type.name

        options_for_patch_datasets = {"type": "OTXClsDataset", "empty_label": self._empty_label}
        options_for_patch_evaluation = {"task": "normal"}
        if self._multilabel:
            options_for_patch_datasets["type"] = "OTXMultilabelClsDataset"
            options_for_patch_evaluation["task"] = "multilabel"
        elif self._hierarchical:
            options_for_patch_datasets["type"] = "OTXHierarchicalClsDataset"
            options_for_patch_datasets["hierarchical_info"] = self._hierarchical_info
            options_for_patch_evaluation["task"] = "hierarchical"
        elif self._selfsl:
            options_for_patch_datasets["type"] = "SelfSLDataset"
        patch_default_config(self._recipe_cfg)
        patch_runner(self._recipe_cfg)
        patch_data_pipeline(self._recipe_cfg, self.data_pipeline_path)
        patch_datasets(
            self._recipe_cfg,
            self._task_type.domain,
            **options_for_patch_datasets,
        )  # for OTX compatibility
        patch_evaluation(self._recipe_cfg, **options_for_patch_evaluation)  # for OTX compatibility

    # TODO: make cfg_path loaded from custom model cfg file corresponding to train_type
    # model.py contains heads/classifier only for Incremental setting
    # error log : ValueError: Unexpected type of 'data_loader' parameter
    def _init_model_cfg(self):
        if self._multilabel:
            cfg_path = os.path.join(self._model_dir, "model_multilabel.py")
        elif self._hierarchical:
            cfg_path = os.path.join(self._model_dir, "model_hierarchical.py")
        else:
            cfg_path = os.path.join(self._model_dir, "model.py")
        cfg = MPAConfig.fromfile(cfg_path)

        cfg.model.multilabel = self._multilabel
        cfg.model.hierarchical = self._hierarchical
        if self._hierarchical:
            cfg.model.head.hierarchical_info = self._hierarchical_info
        return cfg

    def _init_test_data_cfg(self, dataset: DatasetEntity):
        data_cfg = ConfigDict(
            data=ConfigDict(
                train=ConfigDict(
                    otx_dataset=dataset,
                    labels=self._labels,
                ),
                test=ConfigDict(
                    otx_dataset=dataset,
                    labels=self._labels,
                ),
            )
        )
        return data_cfg

    def _update_stage_module(self, stage_module):
        module_prefix = {TrainType.Incremental: "Incr", TrainType.Semisupervised: "SemiSL"}
        if self._train_type in module_prefix and stage_module in ["ClsTrainer", "ClsInferrer"]:
            stage_module = module_prefix[self._train_type] + stage_module
        return stage_module

    def _initialize_post_hook(self, options=None):
        super()._initialize_post_hook(options)
        options["model_builder"] = build_classifier
