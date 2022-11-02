"""Inference Task of OTX ActionClassification."""

# Copyright (C) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import copy
import io
import os
from typing import Iterable, Optional, Tuple

import numpy as np
import torch
from mmaction.datasets import build_dataloader, build_dataset
from mmaction.models import build_model
from mmaction.utils import get_root_logger
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, load_state_dict
from mmcv.utils import Config, ConfigDict

# TODO Replace this by mmaction
from mmdet.parallel import MMDataCPU

# FIXME This is for initialize mmaction adapter
# pylint: disable=unused-import
from otx.algorithms.action.adapters import mmaction as mmaction_adapter
from otx.algorithms.common.tasks.training_base import BaseTask
from otx.algorithms.detection.adapters.mmdet.utils.config_utils import (
    get_data_cfg,
    prepare_for_testing,
    remove_from_config,
    set_data_classes,
)
from otx.algorithms.detection.configs.base import DetectionConfig
from otx.algorithms.detection.utils.utils import InferenceProgressCallback
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.inference_parameters import InferenceParameters
from otx.api.entities.label import Domain
from otx.api.entities.model import (
    ModelEntity,
    ModelFormat,
    ModelOptimizationType,
    ModelPrecision,
)
from otx.api.entities.result_media import ResultMediaEntity
from otx.api.entities.resultset import ResultSetEntity
from otx.api.entities.scored_label import ScoredLabel
from otx.api.entities.task_environment import TaskEnvironment
from otx.api.entities.tensor import TensorEntity
from otx.api.entities.train_parameters import default_progress_callback
from otx.api.serialization.label_mapper import label_schema_to_bytes
from otx.api.usecases.evaluation.metrics_helper import MetricsHelper
from otx.api.usecases.tasks.interfaces.evaluate_interface import IEvaluationTask
from otx.api.usecases.tasks.interfaces.export_interface import ExportType, IExportTask
from otx.api.usecases.tasks.interfaces.inference_interface import IInferenceTask
from otx.api.usecases.tasks.interfaces.unload_interface import IUnload
from otx.api.utils.argument_checks import (
    DatasetParamTypeCheck,
    check_input_parameters_type,
)
from otx.api.utils.vis_utils import get_actmap

logger = get_root_logger()


# pylint: disable=too-many-locals
class ActionClsInferenceTask(BaseTask, IInferenceTask, IExportTask, IEvaluationTask, IUnload):
    """Inference Task Implementation of OTX Action Classification."""

    @check_input_parameters_type()
    def __init__(self, task_environment: TaskEnvironment):
        # self._should_stop = False
        self._model = None
        self.task_environment = task_environment
        super().__init__(DetectionConfig, task_environment)

    @check_input_parameters_type({"dataset": DatasetParamTypeCheck})
    def infer(
        self,
        dataset: DatasetEntity,
        inference_parameters: Optional[InferenceParameters] = None,
    ) -> DatasetEntity:
        """Main infer function of OTX Action Classification."""
        logger.info("infer()")

        if inference_parameters:
            update_progress_callback = inference_parameters.update_progress
        else:
            update_progress_callback = default_progress_callback

        self._time_monitor = InferenceProgressCallback(len(dataset), update_progress_callback)
        # If confidence threshold is adaptive then up-to-date value should be stored in the model
        # and should not be changed during inference. Otherwise user-specified value should be taken.
        if not self._hyperparams.postprocessing.result_based_confidence_threshold:
            self.confidence_threshold = self._hyperparams.postprocessing.confidence_threshold
        logger.info(f"Confidence threshold {self.confidence_threshold}")

        prediction_results, _ = self._infer_detector(dataset, inference_parameters)
        self._add_predictions_to_dataset(prediction_results, dataset, self.confidence_threshold)
        logger.info("Inference completed")
        return dataset

    def _init_task(self, dataset=None, **kwargs):
        # FIXME: Temporary remedy for CVS-88098
        export = kwargs.get("export", False)
        self._initialize(export=export)
        # update model config -> model label schema
        data_classes = [label.name for label in self._labels]
        model_classes = [label.name for label in self._model_label_schema]
        self._model_cfg["model_classes"] = model_classes
        if dataset is not None:
            train_data_cfg = get_data_cfg(self._data_cfg)
            train_data_cfg["data_classes"] = data_classes
            new_classes = np.setdiff1d(data_classes, model_classes).tolist()
            train_data_cfg["new_classes"] = new_classes

        logger.info(f"running task... kwargs = {kwargs}")
        if self._recipe_cfg is None:
            raise RuntimeError("'config' is not initialized yet. call prepare() method before calling this method")

        self._model = self._load_model(self.task_environment.model)

    def _infer_detector(
        self, dataset: DatasetEntity, inference_parameters: Optional[InferenceParameters] = None
    ) -> Tuple[Iterable, float]:
        """Inference wrapper.

        This method triggers the inference and returns `prediction_results` zipped with prediction results,
        feature vectors, and saliency maps. `metric` is returned as a float value if InferenceParameters.is_evaluation
        is set to true, otherwise, `None` is returned.

        Args:
            dataset (DatasetEntity): the validation or test dataset to be inferred with
            inference_parameters (Optional[InferenceParameters], optional): Option to run evaluation or not.
                If `InferenceParameters.is_evaluation=True` then metric is returned, otherwise, both metric and
                saliency maps are empty. Defaults to None.

        Returns:
            Tuple[Iterable, float]: Iterable prediction results for each sample and metric for on the given dataset
        """
        self._data_cfg = self._init_test_data_cfg(dataset)
        if self._recipe_cfg is None:
            self._init_task(dataset)
        if self._recipe_cfg is None:
            raise Exception("Recipe config is not initialized properly")

        dump_features = False
        dump_saliency_map = False

        test_config = prepare_for_testing(self._recipe_cfg, dataset)
        mm_test_dataset = build_dataset(test_config.data.test)
        # TODO Get batch size and num_gpus autometically
        batch_size = 1
        mm_test_dataloader = build_dataloader(
            mm_test_dataset,
            videos_per_gpu=batch_size,
            workers_per_gpu=test_config.data.workers_per_gpu,
            num_gpus=1,
            dist=False,
            shuffle=False,
        )

        self._model.eval()
        if torch.cuda.is_available():
            eval_model = MMDataParallel(self._model.cuda(test_config.gpu_ids[0]), device_ids=test_config.gpu_ids)
        else:
            eval_model = MMDataCPU(self._model)

        eval_predictions = []
        feature_vectors = []
        saliency_maps = []

        def dump_features_hook():
            raise NotImplementedError("get_feature_vector function for mmaction is not implemented")

        # pylint: disable=unused-argument
        def dummy_dump_features_hook(model, inp, out):
            feature_vectors.append(None)

        def dump_saliency_hook():
            raise NotImplementedError("get_saliency_map for mmaction is not implemented")

        # pylint: disable=unused-argument
        def dummy_dump_saliency_hook(model, inp, out):
            saliency_maps.append(None)

        feature_vector_hook = dump_features_hook if dump_features else dummy_dump_features_hook
        saliency_map_hook = dump_saliency_hook if dump_saliency_map else dummy_dump_saliency_hook

        # Use a single gpu for testing. Set in both mm_test_dataloader and eval_model
        with eval_model.module.backbone.register_forward_hook(feature_vector_hook):
            with eval_model.module.backbone.register_forward_hook(saliency_map_hook):
                for data in mm_test_dataloader:
                    with torch.no_grad():
                        result = eval_model(return_loss=False, **data)
                    eval_predictions.extend(result)

        # hard-code way to remove EvalHook args
        for key in ["interval", "tmpdir", "start", "gpu_collect", "save_best", "rule", "dynamic_intervals"]:
            self._recipe_cfg.evaluation.pop(key, None)

        metric = None
        metric_name = self._recipe_cfg.evaluation.final_metric
        if inference_parameters.is_evaluation:
            metric = mm_test_dataset.evaluate(eval_predictions, **self._recipe_cfg.evaluation)[metric_name]

        assert len(eval_predictions) == len(feature_vectors), f"{len(eval_predictions)} != {len(feature_vectors)}"
        assert len(eval_predictions) == len(saliency_maps), f"{len(eval_predictions)} != {len(saliency_maps)}"
        predictions = zip(eval_predictions, feature_vectors, saliency_maps)

        return predictions, metric

    def _load_model(self, model: ModelEntity):
        if self._recipe_cfg is None:
            raise Exception("Recipe config is not initialized properly")
        if model is not None:
            # If a model has been trained and saved for the task already, create empty model and load weights here
            buffer = io.BytesIO(model.get_data("weights.pth"))
            model_data = torch.load(buffer, map_location=torch.device("cpu"))

            self.confidence_threshold = model_data.get("confidence_threshold", self.confidence_threshold)
            model = self._create_model(self._recipe_cfg, from_scratch=True)

            try:
                load_state_dict(model, model_data["model"])

                # It prevent model from being overwritten
                if "load_from" in self._recipe_cfg:
                    self._recipe_cfg.load_from = None

                logger.info("Loaded model weights from Task Environment")
                logger.info(f"Model architecture: {self._model_name}")
                for name, weights in model.named_parameters():
                    if not torch.isfinite(weights).all():
                        logger.info(f"Invalid weights in: {name}. Recreate model from pre-trained weights")
                        model = self._create_model(self._recipe_cfg, from_scratch=False)
                        return model

            except BaseException as ex:
                raise ValueError("Could not load the saved model. The model file structure is invalid.") from ex
        else:
            # If there is no trained model yet, create model with pretrained weights as defined in the model config
            # file.
            model = self._create_model(self._recipe_cfg, from_scratch=False)
            logger.info(
                f"No trained model in project yet. Created new model with '{self._model_name}' "
                f"architecture and general-purpose pretrained weights."
            )
        return model

    @staticmethod
    def _create_model(config: Config, from_scratch: bool = False):
        """Creates a model, based on the configuration in config.

        :param config: mmaction configuration from which the model has to be built
        :param from_scratch: bool, if True does not load any weights

        :return model: ModelEntity in training mode
        """
        model_cfg = copy.deepcopy(config.model)

        init_from = None if from_scratch else config.get("load_from", None)
        logger.warning(init_from)
        if init_from is not None:
            # No need to initialize backbone separately, if all weights are provided.
            # model_cfg.pretrained = None
            logger.warning("build detector")
            model = build_model(model_cfg)
            # Load all weights.
            logger.warning("load checkpoint")
            load_checkpoint(model, init_from, map_location="cpu")
        else:
            logger.warning("build detector")
            model = build_model(model_cfg)
        return model

    @check_input_parameters_type()
    def evaluate(
        self,
        output_resultset: ResultSetEntity,
        evaluation_metric: Optional[str] = None,
    ):
        """Evaluate function of OTX Action Classification Task."""
        logger.info("called evaluate()")
        if evaluation_metric is not None:
            logger.warning(
                f"Requested to use {evaluation_metric} metric, " "but parameter is ignored. Use F-measure instead."
            )
        metric = MetricsHelper.compute_f_measure(output_resultset)
        logger.info(f"F-measure after evaluation: {metric.f_measure.value}")
        output_resultset.performance = metric.get_performance()
        logger.info("Evaluation completed")

    def unload(self):
        """Unload the task."""
        self.finalize()

    @check_input_parameters_type()
    def export(self, export_type: ExportType, output_model: ModelEntity):
        """Export function of OTX Action Classification Task."""
        # copied from OTX inference_task.py
        logger.info("Exporting the model")
        if export_type != ExportType.OPENVINO:
            raise RuntimeError(f"not supported export type {export_type}")
        output_model.model_format = ModelFormat.OPENVINO
        output_model.optimization_type = ModelOptimizationType.MO

        stage_module = "DetectionExporter"
        results = self._run_task(stage_module, mode="train", precision="FP32", export=True)
        results = results.get("outputs")
        logger.debug(f"results of run_task = {results}")
        if results is None:
            logger.error("error while exporting model, result is None")
        else:
            bin_file = results.get("bin")
            xml_file = results.get("xml")
            if xml_file is None or bin_file is None:
                raise RuntimeError("invalid status of exporting. bin and xml should not be None")
            with open(bin_file, "rb") as f:
                output_model.set_data("openvino.bin", f.read())
            with open(xml_file, "rb") as f:
                output_model.set_data("openvino.xml", f.read())
            output_model.set_data(
                "confidence_threshold",
                np.array([self.confidence_threshold], dtype=np.float32).tobytes(),
            )
            output_model.precision = [ModelPrecision.FP32]
            output_model.optimization_methods = self._optimization_methods
            output_model.set_data(
                "label_schema.json",
                label_schema_to_bytes(self._task_environment.label_schema),
            )
        logger.info("Exporting completed")

    def _init_recipe_hparam(self) -> dict:
        configs = super()._init_recipe_hparam()
        configs["use_adaptive_interval"] = self._hyperparams.learning_parameters.use_adaptive_interval
        return configs

    def _init_recipe(self):
        logger.info("called _init_recipe()")
        recipe_root = os.path.abspath(os.path.dirname(self.template_file_path))
        recipe = os.path.join(recipe_root, "model.py")
        self._recipe_cfg = Config.fromfile(recipe)
        # TODO Unify these patch procedure something like patch_config function
        self._patch_data_pipeline()
        self._patch_datasets(self._recipe_cfg)  # for OTX compatibility
        # TODO Handle runner through patch_config function
        self._recipe_cfg.work_dir = self._output_path
        set_data_classes(self._recipe_cfg, self._labels)
        # FIXME This is temporaray solution
        self._recipe_cfg.omnisource = None
        self._recipe_cfg.data.train.start_index = 1
        self._recipe_cfg.data.train.modality = "RGB"
        logger.info(f"initialized recipe = {recipe}")

    def _init_model_cfg(self):
        base_dir = os.path.abspath(os.path.dirname(self.template_file_path))
        model_cfg = Config.fromfile(os.path.join(base_dir, "model.py"))
        return model_cfg

    def _init_test_data_cfg(self, dataset: DatasetEntity):
        data_cfg = ConfigDict(
            data=ConfigDict(
                train=ConfigDict(
                    otx_dataset=None,
                    labels=self._labels,
                ),
                test=ConfigDict(
                    otx_dataset=dataset,
                    labels=self._labels,
                ),
            )
        )
        return data_cfg

    def _add_predictions_to_dataset(self, prediction_results, dataset, confidence_threshold=0.0):
        """Loop over dataset again to assign predictions. Convert from MM format to OTX format."""
        for dataset_item, (all_results, feature_vector, saliency_map) in zip(dataset, prediction_results):
            item_labels = []
            # TODO Check proper label assignment method
            for i, logit in enumerate(all_results):
                if logit > confidence_threshold:
                    label = ScoredLabel(label=self._labels[i], probability=float(logit))
                    item_labels.append(label)
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
                )
                dataset_item.append_metadata_item(saliency_map_media, model=self._task_environment.model)

    def _patch_data_pipeline(self):
        base_dir = os.path.abspath(os.path.dirname(self.template_file_path))
        data_pipeline_path = os.path.join(base_dir, "data_pipeline.py")
        if os.path.exists(data_pipeline_path):
            data_pipeline_cfg = Config.fromfile(data_pipeline_path)
            self._recipe_cfg.merge_from_dict(data_pipeline_cfg)

    @staticmethod
    def _patch_datasets(config: Config):
        # Copied from otx/apis/detection/config_utils.py
        # Added 'unlabeled' data support

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
            cfg.type = "OTXRawframeDataset"
            cfg.otx_dataset = None
            cfg.labels = None
