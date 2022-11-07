# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import io
import os
import time
from collections import defaultdict
from typing import List, Optional

import numpy as np
import torch
from mmcv.utils import ConfigDict
from mpa import MPAConstants
from mpa.stage import Stage
from mpa.utils.config_utils import MPAConfig
from mpa.utils.logger import get_logger
from mpa_tasks.apis import BaseTask, TrainType
from mpa_tasks.apis.classification import ClassificationConfig
from ote_sdk.configuration import cfg_helper
from ote_sdk.configuration.helper.utils import ids_to_strings
from ote_sdk.entities.datasets import DatasetEntity
from ote_sdk.entities.inference_parameters import (
    InferenceParameters,
    default_progress_callback,
)
from ote_sdk.entities.label import Domain
from ote_sdk.entities.metrics import (
    CurveMetric,
    LineChartInfo,
    LineMetricsGroup,
    MetricsGroup,
    Performance,
    ScoreMetric,
)
from ote_sdk.entities.model import (  # ModelStatus
    ModelEntity,
    ModelFormat,
    ModelOptimizationType,
    ModelPrecision,
)
from ote_sdk.entities.model_template import parse_model_template
from ote_sdk.entities.result_media import ResultMediaEntity
from ote_sdk.entities.resultset import ResultSetEntity
from ote_sdk.entities.scored_label import ScoredLabel
from ote_sdk.entities.subset import Subset
from ote_sdk.entities.task_environment import TaskEnvironment
from ote_sdk.entities.tensor import TensorEntity
from ote_sdk.entities.train_parameters import TrainParameters, UpdateProgressCallback
from ote_sdk.entities.train_parameters import (
    default_progress_callback as train_default_progress_callback,
)
from ote_sdk.serialization.label_mapper import label_schema_to_bytes
from ote_sdk.usecases.evaluation.metrics_helper import MetricsHelper
from ote_sdk.usecases.reporting.time_monitor_callback import TimeMonitorCallback
from ote_sdk.usecases.tasks.interfaces.evaluate_interface import IEvaluationTask
from ote_sdk.usecases.tasks.interfaces.explain_interface import IExplainTask
from ote_sdk.usecases.tasks.interfaces.export_interface import ExportType, IExportTask
from ote_sdk.usecases.tasks.interfaces.inference_interface import IInferenceTask
from ote_sdk.usecases.tasks.interfaces.unload_interface import IUnload
from ote_sdk.utils.argument_checks import check_input_parameters_type
from ote_sdk.utils.labels_utils import get_empty_label
from ote_sdk.utils.vis_utils import get_actmap
from torchreid_tasks.nncf_task import OTEClassificationNNCFTask

# from torchreid_tasks.utils import TrainingProgressCallback
from torchreid_tasks.utils import OTELoggerHook
from torchreid_tasks.utils import get_multihead_class_info as get_hierarchical_info

logger = get_logger()

TASK_CONFIG = ClassificationConfig


class TrainingProgressCallback(TimeMonitorCallback):
    def __init__(self, update_progress_callback: UpdateProgressCallback):
        super().__init__(0, 0, 0, 0, update_progress_callback=update_progress_callback)

    def on_train_batch_end(self, batch, logs=None):
        super().on_train_batch_end(batch, logs)
        self.update_progress_callback(self.get_progress())

    def on_epoch_end(self, epoch, logs=None):
        self.past_epoch_duration.append(time.time() - self.start_epoch_time)
        self._calculate_average_epoch()
        score = None
        if hasattr(self.update_progress_callback, "metric") and isinstance(logs, dict):
            score = logs.get(self.update_progress_callback.metric, None)
            logger.info(f"logged score for metric {self.update_progress_callback.metric} = {score}")
            score = 0.01 * float(score) if score is not None else None
            if score is not None:
                iter_num = logs.get("current_iters", None)
                if iter_num is not None:
                    logger.info(f"score = {score} at epoch {epoch} / {int(iter_num)}")
                    # as a trick, score (at least if it's accuracy not the loss) and iteration number
                    # could be assembled just using summation and then disassembeled.
                    if 1.0 > score:
                        score = score + int(iter_num)
                    else:
                        score = -(score + int(iter_num))
        self.update_progress_callback(self.get_progress(), score=score)


class ClassificationInferenceTask(BaseTask, IInferenceTask, IExportTask, IEvaluationTask, IExplainTask, IUnload):
    def __init__(self, task_environment: TaskEnvironment):
        self._should_stop = False
        super().__init__(TASK_CONFIG, task_environment)

        self._task_environment = task_environment
        if len(task_environment.get_labels(False)) == 1:
            self._labels = task_environment.get_labels(include_empty=True)
        else:
            self._labels = task_environment.get_labels(include_empty=False)
        self._empty_label = get_empty_label(task_environment.label_schema)
        self._multilabel = False
        self._hierarchical = False

        self._multilabel = len(task_environment.label_schema.get_groups(False)) > 1 and len(
            task_environment.label_schema.get_groups(False)
        ) == len(
            task_environment.get_labels(include_empty=False)
        )  # noqa:E127

        self._hierarchical_info = None
        if not self._multilabel and len(task_environment.label_schema.get_groups(False)) > 1:
            self._hierarchical = True
            self._hierarchical_info = get_hierarchical_info(task_environment.label_schema)

    def infer(
        self,
        dataset: DatasetEntity,
        inference_parameters: Optional[InferenceParameters] = None,
    ) -> DatasetEntity:
        logger.info("called infer()")
        stage_module = "ClsInferrer"
        self._data_cfg = self._init_test_data_cfg(dataset)

        dump_features = True
        dump_saliency_map = not inference_parameters.is_evaluation if inference_parameters else True
        results = self._run_task(
            stage_module,
            mode="train",
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
            update_progress_callback = inference_parameters.update_progress

        self._add_predictions_to_dataset(prediction_results, dataset, update_progress_callback)
        return dataset

    def explain(
        self,
        dataset: DatasetEntity,
        explain_parameters: Optional[InferenceParameters] = None,
    ) -> DatasetEntity:
        logger.info("called explain()")
        stage_module = "ClsExplainer"
        self._data_cfg = self._init_test_data_cfg(dataset)
        dataset = dataset.with_empty_annotations()

        results = self._run_task(
            stage_module,
            mode="train",
            dataset=dataset,
            explainer=explain_parameters.explainer,
        )
        logger.debug(f"result of run_task {stage_module} module = {results}")
        saliency_maps = results["outputs"]["saliency_maps"]
        update_progress_callback = default_progress_callback
        if explain_parameters is not None:
            update_progress_callback = explain_parameters.update_progress

        self._add_saliency_maps_to_dataset(saliency_maps, dataset, update_progress_callback)
        return dataset

    def evaluate(
        self,
        output_result_set: ResultSetEntity,
        evaluation_metric: Optional[str] = None,
    ):
        logger.info("called evaluate()")
        metric = MetricsHelper.compute_accuracy(output_result_set)
        logger.info(f"Accuracy after evaluation: {metric.accuracy.value}")
        output_result_set.performance = metric.get_performance()
        logger.info("Evaluation completed")

    def unload(self):
        logger.info("called unload()")
        self.finalize()

    def export(self, export_type: ExportType, output_model: ModelEntity):
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
            logger.error(f"error while exporting model {results.get('msg')}")
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

    def _add_predictions_to_dataset(self, prediction_results, dataset, update_progress_callback):
        """Loop over dataset again to assign predictions. Convert from MMClassification format to OTE format."""
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
                    ote_label = next(x for x in self._labels if x.name == label_str)
                    item_labels.append(ScoredLabel(label=ote_label, probability=float(head_logits[head_pred])))

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
                            ote_label = next(x for x in self._labels if x.name == label_str)
                            item_labels.append(ScoredLabel(label=ote_label, probability=float(logit)))
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
                saliency_map = get_actmap(saliency_map, (dataset_item.width, dataset_item.height))
                saliency_map_media = ResultMediaEntity(
                    name="Saliency Map",
                    type="saliency_map",
                    annotation_scene=dataset_item.annotation_scene,
                    numpy=saliency_map,
                    roi=dataset_item.roi,
                    label=item_labels[0].label,
                )
                dataset_item.append_metadata_item(saliency_map_media, model=self._task_environment.model)

            update_progress_callback(int(i / dataset_size * 100))

    def _add_saliency_maps_to_dataset(self, saliency_maps, dataset, update_progress_callback):
        """Loop over dataset again and assign activation maps"""
        dataset_size = len(dataset)
        for i, (dataset_item, saliency_map) in enumerate(zip(dataset, saliency_maps)):
            saliency_map = get_actmap(saliency_map, (dataset_item.width, dataset_item.height))
            saliency_map_media = ResultMediaEntity(
                name="Saliency Map",
                type="saliency_map",
                annotation_scene=dataset_item.annotation_scene,
                numpy=saliency_map,
                roi=dataset_item.roi,
            )
            dataset_item.append_metadata_item(saliency_map_media, model=self._task_environment.model)
            update_progress_callback(int(i / dataset_size * 100))

    def _init_recipe(self):
        logger.info("called _init_recipe()")

        recipe_root = os.path.join(MPAConstants.RECIPES_PATH, "stages/classification")
        train_type = self._hyperparams.algo_backend.train_type
        logger.info(f"train type = {train_type}")
        recipe = os.path.join(recipe_root, "class_incr.yaml")

        if train_type == TrainType.SemiSupervised:
            raise NotImplementedError(f"train type {train_type} is not implemented yet.")
        elif train_type == TrainType.SelfSupervised:
            raise NotImplementedError(f"train type {train_type} is not implemented yet.")
        elif train_type == TrainType.Incremental and self._multilabel:
            recipe = os.path.join(recipe_root, "class_incr_multilabel.yaml")
        else:
            # raise NotImplementedError(f'train type {train_type} is not implemented yet.')
            # FIXME: Temporary remedy for CVS-88098
            logger.warning(f"train type {train_type} is not implemented yet. Running incremental training.")

        self._recipe_cfg = MPAConfig.fromfile(recipe)
        self._patch_datasets(self._recipe_cfg)  # for OTE compatibility
        self._patch_evaluation(self._recipe_cfg)  # for OTE compatibility
        logger.info(f"initialized recipe = {recipe}")

    def _init_model_cfg(self):
        base_dir = os.path.abspath(os.path.dirname(self.template_file_path))
        if self._multilabel:
            cfg_path = os.path.join(base_dir, "model_multilabel.py")
        elif self._hierarchical:
            cfg_path = os.path.join(base_dir, "model_hierarchical.py")
        else:
            cfg_path = os.path.join(base_dir, "model.py")
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
                    ote_dataset=dataset,
                    labels=self._labels,
                ),
                test=ConfigDict(
                    ote_dataset=dataset,
                    labels=self._labels,
                ),
            )
        )
        return data_cfg

    def _overwrite_parameters(self):
        super()._overwrite_parameters()
        if self._multilabel:
            # hack to use 1cycle policy
            self._recipe_cfg.merge_from_dict(
                ConfigDict(
                    lr_config=ConfigDict(max_lr=self._hyperparams.learning_parameters.learning_rate, warmup=None)
                )
            )

    def _patch_datasets(self, config: MPAConfig, domain=Domain.CLASSIFICATION):
        def patch_color_conversion(pipeline):
            # Default data format for OTE is RGB, while mmdet uses BGR, so negate the color conversion flag.
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
        for subset in ("train", "val", "test"):
            cfg = config.data.get(subset, None)
            if not cfg:
                continue
            if cfg.type == "RepeatDataset":
                cfg = cfg.dataset

            if self._multilabel:
                cfg.type = "MPAMultilabelClsDataset"
            elif self._hierarchical:
                cfg.type = "MPAHierarchicalClsDataset"
                cfg.hierarchical_info = self._hierarchical_info
                if subset == "train":
                    cfg.drop_last = True  # For stable hierarchical information indexing
            else:
                cfg.type = "MPAClsDataset"

            # In train dataset, when sample size is smaller than batch size
            if subset == "train" and self._data_cfg:
                train_data_cfg = Stage.get_data_cfg(self._data_cfg, "train")
                if len(train_data_cfg.get("ote_dataset", [])) < self._recipe_cfg.data.get("samples_per_gpu", 2):
                    cfg.drop_last = False

            cfg.domain = domain
            cfg.ote_dataset = None
            cfg.labels = None
            cfg.empty_label = self._empty_label
            for pipeline_step in cfg.pipeline:
                if subset == "train" and pipeline_step.type == "Collect":
                    pipeline_step = BaseTask._get_meta_keys(pipeline_step)
            patch_color_conversion(cfg.pipeline)

    def _patch_evaluation(self, config: MPAConfig):
        cfg = config.evaluation
        if self._multilabel:
            cfg.metric = ["accuracy-mlc", "mAP", "CP", "OP", "CR", "OR", "CF1", "OF1"]
            config.early_stop_metric = "mAP"
        elif self._hierarchical:
            cfg.metric = ["MHAcc", "avgClsAcc", "mAP"]
            config.early_stop_metric = "MHAcc"
        else:
            cfg.metric = ["accuracy", "class_accuracy"]
            config.early_stop_metric = "accuracy"


class ClassificationTrainTask(ClassificationInferenceTask):
    def save_model(self, output_model: ModelEntity):
        logger.info("called save_model")
        buffer = io.BytesIO()
        hyperparams_str = ids_to_strings(cfg_helper.convert(self._hyperparams, dict, enum_to_str=True))
        labels = {label.name: label.color.rgb_tuple for label in self._labels}
        model_ckpt = torch.load(self._model_ckpt)
        modelinfo = {
            "model": model_ckpt["state_dict"],
            "config": hyperparams_str,
            "labels": labels,
            "VERSION": 1,
        }

        torch.save(modelinfo, buffer)
        output_model.set_data("weights.pth", buffer.getvalue())
        output_model.set_data(
            "label_schema.json",
            label_schema_to_bytes(self._task_environment.label_schema),
        )
        output_model.precision = self._precision

    def cancel_training(self):
        """
        Sends a cancel training signal to gracefully stop the optimizer. The signal consists of creating a
        '.stop_training' file in the current work_dir. The runner checks for this file periodically.
        The stopping mechanism allows stopping after each iteration, but validation will still be carried out. Stopping
        will therefore take some time.
        """
        self._should_stop = True
        logger.info("Cancel training requested.")
        if self.cancel_interface is not None:
            self.cancel_interface.cancel()
        else:
            logger.info("but training was not started yet. reserved it to cancel")
            self.reserved_cancel = True

    def train(
        self,
        dataset: DatasetEntity,
        output_model: ModelEntity,
        train_parameters: Optional[TrainParameters] = None,
    ):
        logger.info("train()")
        # Check for stop signal between pre-eval and training.
        # If training is cancelled at this point,
        if self._should_stop:
            logger.info("Training cancelled.")
            self._should_stop = False
            self._is_training = False
            return

        # Set OTE LoggerHook & Time Monitor
        update_progress_callback = train_default_progress_callback
        if train_parameters is not None:
            update_progress_callback = train_parameters.update_progress
        self._time_monitor = TrainingProgressCallback(update_progress_callback)
        self._learning_curves = defaultdict(OTELoggerHook.Curve)

        stage_module = "ClsTrainer"
        self._data_cfg = self._init_train_data_cfg(dataset)
        self._is_training = True
        results = self._run_task(stage_module, mode="train", dataset=dataset, parameters=train_parameters)

        # Check for stop signal between pre-eval and training.
        # If training is cancelled at this point,
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
        else:
            # update checkpoint to the newly trained model
            self._model_ckpt = model_ckpt

        # compose performance statistics
        training_metrics, final_acc = self._generate_training_metrics_group(self._learning_curves)
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

    def _init_train_data_cfg(self, dataset: DatasetEntity):
        logger.info("init data cfg.")
        data_cfg = ConfigDict(
            data=ConfigDict(
                train=ConfigDict(
                    ote_dataset=dataset.get_subset(Subset.TRAINING),
                    labels=self._labels,
                    label_names=list(label.name for label in self._labels),
                ),
                val=ConfigDict(
                    ote_dataset=dataset.get_subset(Subset.VALIDATION),
                    labels=self._labels,
                ),
            )
        )

        for label in self._labels:
            label.hotkey = "a"
        return data_cfg

    def _generate_training_metrics_group(self, learning_curves) -> Optional[List[MetricsGroup]]:
        """
        Parses the classification logs to get metrics from the latest training run
        :return output List[MetricsGroup]
        """
        output: List[MetricsGroup] = []

        if self._multilabel:
            metric_key = "val/accuracy-mlc"
        elif self._hierarchical:
            metric_key = "val/MHAcc"
        else:
            metric_key = "val/accuracy_top-1"

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


class ClassificationNNCFTask(OTEClassificationNNCFTask):
    @check_input_parameters_type()
    def __init__(self, task_environment: TaskEnvironment):
        """ "
        Task for compressing classification models using NNCF.
        """
        curr_model_path = task_environment.model_template.model_template_path
        base_model_path = os.path.join(
            os.path.dirname(os.path.abspath(curr_model_path)),
            task_environment.model_template.base_model_path,
        )
        if os.path.isfile(base_model_path):
            logger.info(f"Base model for NNCF: {base_model_path}")
            # Redirect to base model
            task_environment.model_template = parse_model_template(base_model_path)
        super().__init__(task_environment)
