"""Inference Task of OTX Classification."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os
from typing import Optional

import numpy as np
from mmcv.utils import ConfigDict

from otx.algorithms.classification.configs import ClassificationConfig
from otx.algorithms.classification.utils import (
    get_multihead_class_info as get_hierarchical_info,
)
from otx.algorithms.common.adapters.mmcv.utils import get_meta_keys, patch_data_pipeline
from otx.algorithms.common.configs import TrainType
from otx.algorithms.common.tasks import BaseTask
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.inference_parameters import (
    InferenceParameters,
    default_progress_callback,
)
from otx.api.entities.label import Domain
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
from otx.api.utils.dataset_utils import add_saliency_maps_to_dataset_item
from otx.api.utils.labels_utils import get_empty_label
from otx.mpa import MPAConstants
from otx.mpa.stage import Stage
from otx.mpa.utils.config_utils import MPAConfig
from otx.mpa.utils.logger import get_logger

# pylint: disable=invalid-name

logger = get_logger()

TASK_CONFIG = ClassificationConfig
RECIPE_TRAIN_TYPE = {
    TrainType.SEMISUPERVISED: "semi.yaml",
    TrainType.INCREMENTAL: "incremental.yaml",
    TrainType.SELFSUPERVISED: "selfsl.yaml",
}


class ClassificationInferenceTask(
    BaseTask, IInferenceTask, IExportTask, IEvaluationTask, IExplainTask, IUnload
):  # pylint: disable=too-many-instance-attributes
    """Inference Task Implementation of OTX Classification."""

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

        self._hierarchical_info = None
        if not self._multilabel and len(task_environment.label_schema.get_groups(False)) > 1:
            self._hierarchical = True
            self._hierarchical_info = get_hierarchical_info(task_environment.label_schema)

        if self._hyperparams.algo_backend.train_type == TrainType.SELFSUPERVISED:
            self._selfsl = True

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
        if inference_parameters is not None:
            update_progress_callback = inference_parameters.update_progress  # type: ignore

        self._add_predictions_to_dataset(prediction_results, dataset, update_progress_callback)
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
        saliency_maps = results["outputs"]["saliency_maps"]
        update_progress_callback = default_progress_callback
        if explain_parameters is not None:
            update_progress_callback = explain_parameters.update_progress  # type: ignore

        self._add_saliency_maps_to_dataset(saliency_maps, dataset, update_progress_callback)
        return dataset

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
        self.finalize()

    def export(self, export_type: ExportType, output_model: ModelEntity):
        """Export function of OTX Classification Task."""

        logger.info("Exporting the model")
        if export_type != ExportType.OPENVINO:
            raise RuntimeError(f"not supported export type {export_type}")
        output_model.model_format = ModelFormat.OPENVINO
        output_model.optimization_type = ModelOptimizationType.MO

        stage_module = "ClsExporter"
        results = self._run_task(stage_module, mode="train", precision="FP32", export=True)
        logger.debug(f"results of run_task = {results}")
        results = results.get("outputs")
        logger.debug(f"results of run_task = {results}")
        if results is None:
            logger.error("Error while exporting model. Result is NoneType.")
        else:
            bin_file = results.get("bin")
            xml_file = results.get("xml")
            if xml_file is None or bin_file is None:
                raise RuntimeError("invalid status of exporting. bin and xml should not be None")
            with open(bin_file, "rb") as f:
                output_model.set_data("openvino.bin", f.read())
            with open(xml_file, "rb") as f:
                output_model.set_data("openvino.xml", f.read())
        output_model.precision = [ModelPrecision.FP32]
        output_model.set_data(
            "label_schema.json",
            label_schema_to_bytes(self._task_environment.label_schema),
        )
        logger.info("Exporting completed")

    # pylint: disable=too-many-branches, too-many-locals
    def _add_predictions_to_dataset(self, prediction_results, dataset, update_progress_callback):
        """Loop over dataset again to assign predictions.Convert from MMClassification format to OTX format."""

        dataset_size = len(dataset)
        for i, (dataset_item, prediction_items) in enumerate(zip(dataset, prediction_results)):
            item_labels = []
            pos_thr = 0.5
            prediction_item, feature_vector, saliency_map = prediction_items
            if any(np.isnan(prediction_item)):
                logger.info("Nan in prediction_item.")

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
                    logits_begin, logits_end = self._hierarchical_info["head_idx_to_logits_range"][head_idx]
                    head_logits = prediction_item[logits_begin:logits_end]
                    head_pred = np.argmax(head_logits)  # Assume logits already passed softmax
                    label_str = self._hierarchical_info["all_groups"][head_idx][head_pred]
                    otx_label = next(x for x in self._labels if x.name == label_str)
                    item_labels.append(ScoredLabel(label=otx_label, probability=float(head_logits[head_pred])))

                if self._hierarchical_info["num_multilabel_classes"]:
                    logits_begin, logits_end = (
                        self._hierarchical_info["num_single_label_classes"],
                        -1,
                    )
                    head_logits = prediction_item[logits_begin:logits_end]
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

            dataset_item.append_labels(item_labels)

            if feature_vector is not None:
                active_score = TensorEntity(name="representation_vector", numpy=feature_vector.reshape(-1))
                dataset_item.append_metadata_item(active_score, model=self._task_environment.model)

            if saliency_map is not None:
                add_saliency_maps_to_dataset_item(
                    dataset_item=dataset_item,
                    saliency_map=saliency_map,
                    model=self._task_environment.model,
                    labels=self._labels,
                    task="cls",
                )
            update_progress_callback(int(i / dataset_size * 100))

    def _add_saliency_maps_to_dataset(self, saliency_maps, dataset, update_progress_callback):
        """Loop over dataset again and assign saliency maps."""
        dataset_size = len(dataset)
        for i, (dataset_item, saliency_map) in enumerate(zip(dataset, saliency_maps)):
            add_saliency_maps_to_dataset_item(
                dataset_item=dataset_item,
                saliency_map=saliency_map,
                model=self._task_environment.model,
                labels=self._labels,
                task="cls",
            )
            update_progress_callback(int(i / dataset_size * 100))

    def _init_recipe_hparam(self) -> dict:
        warmup_iters = int(self._hyperparams.learning_parameters.learning_rate_warmup_iters)
        if self._multilabel:
            # hack to use 1cycle policy
            lr_config = ConfigDict(max_lr=self._hyperparams.learning_parameters.learning_rate, warmup=None)
        else:
            lr_config = (
                ConfigDict(warmup_iters=warmup_iters) if warmup_iters > 0 else ConfigDict(warmup_iters=0, warmup=None)
            )

        if self._hyperparams.learning_parameters.enable_early_stopping:
            early_stop = ConfigDict(
                start=int(self._hyperparams.learning_parameters.early_stop_start),
                patience=int(self._hyperparams.learning_parameters.early_stop_patience),
                iteration_patience=int(self._hyperparams.learning_parameters.early_stop_iteration_patience),
            )
        else:
            early_stop = False

        if self._recipe_cfg.runner.get("type") == "IterBasedRunner":  # type: ignore
            runner = ConfigDict(max_iters=int(self._hyperparams.learning_parameters.num_iters))
        else:
            runner = ConfigDict(max_epochs=int(self._hyperparams.learning_parameters.num_iters))

        return ConfigDict(
            optimizer=ConfigDict(lr=self._hyperparams.learning_parameters.learning_rate),
            lr_config=lr_config,
            early_stop=early_stop,
            data=ConfigDict(
                samples_per_gpu=int(self._hyperparams.learning_parameters.batch_size),
                workers_per_gpu=int(self._hyperparams.learning_parameters.num_workers),
            ),
            runner=runner,
        )

    def _init_recipe(self):
        logger.info("called _init_recipe()")

        if self._multilabel:
            recipe_root = os.path.join(MPAConstants.RECIPES_PATH, "stages/classification/multilabel")
        else:
            recipe_root = os.path.join(MPAConstants.RECIPES_PATH, "stages/classification")

        logger.info(f"train type = {self._train_type}")

        if self._train_type in RECIPE_TRAIN_TYPE:
            recipe = os.path.join(recipe_root, RECIPE_TRAIN_TYPE[self._train_type])
        else:
            raise NotImplementedError(f"Train type {self._train_type} is not implemented yet.")

        logger.info(f"train type = {self._train_type} - loading {recipe}")

        self._recipe_cfg = MPAConfig.fromfile(recipe)

        # FIXME[Soobee] : if train type is not in cfg, it raises an error in default INCREMENTAL mode.
        # During semi-implementation, this line should be fixed to -> self._recipe_cfg.train_type = train_type
        self._recipe_cfg.train_type = self._train_type.name
        patch_data_pipeline(self._recipe_cfg, self.data_pipeline_path)
        self._patch_datasets(self._recipe_cfg)  # for OTX compatibility
        self._patch_evaluation(self._recipe_cfg)  # for OTX compatibility
        logger.info(f"initialized recipe = {recipe}")

    # TODO: make cfg_path loaded from custom model cfg file corresponding to train_type
    # model.py contains heads/classifier only for INCREMENTAL setting
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

    def _patch_datasets(self, config: MPAConfig, domain=Domain.CLASSIFICATION):  # noqa: C901
        def patch_color_conversion(pipeline):
            # Default data format for OTX is RGB, while mmdet uses BGR, so negate the color conversion flag.
            for pipeline_step in pipeline:
                if pipeline_step.type == "Normalize":
                    to_rgb = False
                    if "to_rgb" in pipeline_step:
                        to_rgb = pipeline_step.to_rgb
                    to_rgb = not bool(to_rgb)
                    pipeline_step.to_rgb = to_rgb
                elif pipeline_step.type == "MultiScaleFlipAug":
                    patch_color_conversion(pipeline_step.transforms)

        assert "data" in config
        for subset in ("train", "val", "test", "unlabeled"):
            cfg = config.data.get(subset, None)
            if not cfg:
                continue
            if cfg.type == "RepeatDataset":
                cfg = cfg.dataset

            else:
                if self._multilabel:
                    cfg.type = "MPAMultilabelClsDataset"
                elif self._hierarchical:
                    cfg.type = "MPAHierarchicalClsDataset"
                    cfg.hierarchical_info = self._hierarchical_info
                    if subset == "train":
                        cfg.drop_last = True  # For stable hierarchical information indexing
                elif self._selfsl:
                    cfg.type = "SelfSLDataset"
                else:
                    cfg.type = "MPAClsDataset"

            # In train dataset, when sample size is smaller than batch size
            if subset == "train" and self._data_cfg:
                train_data_cfg = Stage.get_data_cfg(self._data_cfg, "train")
                if len(train_data_cfg.get("otx_dataset", [])) < self._recipe_cfg.data.get("samples_per_gpu", 2):
                    cfg.drop_last = False

            cfg.domain = domain
            cfg.otx_dataset = None
            cfg.labels = None
            cfg.empty_label = self._empty_label
            for pipeline_step in cfg.pipeline:
                if self._selfsl:
                    # TODO : refactoring
                    # SelfSLDataset has pipelines={"view0": [...], "view1": [...]}.
                    # To access `Normalize`, patch_color_conversion must be applied to both view0 and view1.
                    for pipeline_view in cfg.pipeline[pipeline_step]:
                        if subset == "train" and pipeline_view.type == "Collect":
                            pipeline_view = get_meta_keys(pipeline_view)
                    patch_color_conversion(cfg.pipeline[pipeline_step])
                else:
                    if subset == "train" and pipeline_step.type == "Collect":
                        pipeline_step = get_meta_keys(pipeline_step)

            if not self._selfsl:
                patch_color_conversion(cfg.pipeline)

    def _patch_evaluation(self, config: MPAConfig):
        cfg = config.get("evaluation", None)
        if cfg:
            if self._multilabel:
                cfg.metric = ["accuracy-mlc", "mAP", "CP", "OP", "CR", "OR", "CF1", "OF1"]
                config.early_stop_metric = "mAP"
            elif self._hierarchical:
                cfg.metric = ["MHAcc", "avgClsAcc", "mAP"]
                config.early_stop_metric = "MHAcc"
            else:
                cfg.metric = ["accuracy", "class_accuracy"]
                config.early_stop_metric = "accuracy"
