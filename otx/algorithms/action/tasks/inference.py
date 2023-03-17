"""Inference Task of OTX Action Task."""

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
import warnings
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
from mmaction.datasets import build_dataloader, build_dataset
from mmaction.models import build_model
from mmaction.utils import get_root_logger
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, load_state_dict
from mmcv.utils import Config

from otx.algorithms.action.adapters.mmaction import (
    Exporter,
    patch_config,
    set_data_classes,
)
from otx.algorithms.action.configs.base import ActionConfig
from otx.algorithms.common.adapters.mmcv.utils import prepare_for_testing
from otx.algorithms.common.tasks.training_base import BaseTask
from otx.algorithms.common.utils.callback import InferenceProgressCallback
from otx.api.entities.annotation import Annotation
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.inference_parameters import InferenceParameters
from otx.api.entities.model import (
    ModelEntity,
    ModelFormat,
    ModelOptimizationType,
    ModelPrecision,
)
from otx.api.entities.model_template import TaskType
from otx.api.entities.result_media import ResultMediaEntity
from otx.api.entities.resultset import ResultSetEntity
from otx.api.entities.scored_label import ScoredLabel
from otx.api.entities.shapes.rectangle import Rectangle
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


# pylint: disable=too-many-locals, unused-argument
class ActionInferenceTask(BaseTask, IInferenceTask, IExportTask, IEvaluationTask, IUnload):
    """Inference Task Implementation of OTX Action Task."""

    @check_input_parameters_type()
    def __init__(self, task_environment: TaskEnvironment, **kwargs):
        super().__init__(ActionConfig, task_environment, **kwargs)
        self.deploy_cfg = None

    @check_input_parameters_type({"dataset": DatasetParamTypeCheck})
    def infer(
        self,
        dataset: DatasetEntity,
        inference_parameters: Optional[InferenceParameters] = None,
    ) -> DatasetEntity:
        """Main infer function of OTX Action Task."""
        logger.info("infer()")

        if inference_parameters:
            update_progress_callback = inference_parameters.update_progress
        else:
            update_progress_callback = default_progress_callback

        self._time_monitor = InferenceProgressCallback(len(dataset), update_progress_callback)

        def pre_hook(module, inp):
            self._time_monitor.on_test_batch_begin(None, None)

        def hook(module, inp, out):
            self._time_monitor.on_test_batch_end(None, None)

        if self._recipe_cfg is None:
            self._init_task()
        if self._model:
            with self._model.register_forward_pre_hook(pre_hook), self._model.register_forward_hook(hook):
                prediction_results, _ = self._infer_model(dataset, inference_parameters)
            # TODO Load _add_predictions_to_dataset function from self._task_type
            if self._task_type == TaskType.ACTION_CLASSIFICATION:
                self._add_predictions_to_dataset(prediction_results, dataset)
            elif self._task_type == TaskType.ACTION_DETECTION:
                self._add_det_predictions_to_dataset(prediction_results, dataset)
            logger.info("Inference completed")
        else:
            raise Exception("Model initialization is failed")
        return dataset

    def _initialize_post_hook(self, options=None):
        """Procedure after inialization."""

        if options is None:
            return

        if "deploy_cfg" in options:
            self.deploy_cfg = options["deploy_cfg"]

    def _infer_model(
        self, dataset: DatasetEntity, inference_parameters: Optional[InferenceParameters] = None
    ) -> Tuple[Iterable, Optional[float]]:
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
            eval_model = MMDataParallel(self._model)

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
        if inference_parameters:
            if inference_parameters.is_evaluation:
                metric = mm_test_dataset.evaluate(eval_predictions, **self._recipe_cfg.evaluation)[metric_name]

        assert len(eval_predictions) == len(feature_vectors), f"{len(eval_predictions)} != {len(feature_vectors)}"
        assert len(eval_predictions) == len(saliency_maps), f"{len(eval_predictions)} != {len(saliency_maps)}"
        predictions = zip(eval_predictions, feature_vectors, saliency_maps)

        return predictions, metric

    # pylint: disable=attribute-defined-outside-init
    def _init_task(self, **kwargs):
        # FIXME: Temporary remedy for CVS-88098
        self._initialize(kwargs)
        logger.info(f"running task... kwargs = {kwargs}")
        if self._recipe_cfg is None:
            raise RuntimeError("'config' is not initialized yet. call prepare() method before calling this method")

        self._model = self._load_model(self._task_environment.model)

    def _load_model(self, model: ModelEntity):
        if self._recipe_cfg is None:
            raise Exception("Recipe config is not initialized properly")
        if model is not None:
            # If a model has been trained and saved for the task already, create empty model and load weights here
            buffer = io.BytesIO(model.get_data("weights.pth"))
            model_data = torch.load(buffer, map_location=torch.device("cpu"))

            self.confidence_threshold: float = model_data.get("confidence_threshold", self.confidence_threshold)
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
            logger.warning("build model")
            model = build_model(model_cfg)
            # Load all weights.
            logger.warning("load checkpoint")
            load_checkpoint(model, init_from, map_location="cpu")
        else:
            logger.warning("build model")
            model = build_model(model_cfg)
        return model

    @check_input_parameters_type()
    def evaluate(
        self,
        output_resultset: ResultSetEntity,
    ):
        """Evaluate function of OTX Action Task."""
        logger.info("called evaluate()")
        self._remove_empty_frames(output_resultset.ground_truth_dataset)
        metric = self._get_metric(output_resultset)
        performance = metric.get_performance()
        logger.info(f"Final model performance: {str(performance)}")
        output_resultset.performance = metric.get_performance()
        logger.info("Evaluation completed")

    def _get_metric(self, output_resultset: ResultSetEntity):
        if self._task_type == TaskType.ACTION_CLASSIFICATION:
            return MetricsHelper.compute_accuracy(output_resultset)
        if self._task_type == TaskType.ACTION_DETECTION:
            return MetricsHelper.compute_f_measure(output_resultset)
        raise NotImplementedError(f"{self._task_type} is not supported in action task")

    def _remove_empty_frames(self, dataset: DatasetEntity):
        """Remove empty frame for action detection dataset."""
        remove_indices = []
        for idx, item in enumerate(dataset):
            if item.get_metadata()[0].data.is_empty_frame:
                remove_indices.append(idx)
        dataset.remove_at_indices(remove_indices)

    def unload(self):
        """Unload the task."""
        if self._work_dir_is_temp:
            self._delete_scratch_space()

    @check_input_parameters_type()
    def export(
        self,
        export_type: ExportType,
        output_model: ModelEntity,
        precision: ModelPrecision = ModelPrecision.FP32,
        dump_features: bool = False,
    ):
        """Export function of OTX Action Task."""
        if dump_features:
            raise NotImplementedError(
                "Feature dumping is not implemented for the anomaly task."
                "The saliency maps and representation vector outputs will not be dumped in the exported model."
            )

        # copied from OTX inference_task.py
        logger.info("Exporting the model")
        if export_type != ExportType.OPENVINO:
            raise RuntimeError(f"not supported export type {export_type}")
        output_model.model_format = ModelFormat.OPENVINO
        output_model.optimization_type = ModelOptimizationType.MO
        self._init_task(export=True, dump_features=dump_features)

        self._precision[0] = precision
        half_precision = precision == ModelPrecision.FP16

        try:
            from torch.jit._trace import TracerWarning

            warnings.filterwarnings("ignore", category=TracerWarning)
            exporter = Exporter(
                self._recipe_cfg,
                self._model.state_dict(),
                self.deploy_cfg,
                f"{self._output_path}/openvino",
                half_precision,
            )
            exporter.export()
            bin_file = [f for f in os.listdir(self._output_path) if f.endswith(".bin")][0]
            xml_file = [f for f in os.listdir(self._output_path) if f.endswith(".xml")][0]
            with open(os.path.join(self._output_path, bin_file), "rb") as f:
                output_model.set_data("openvino.bin", f.read())
            with open(os.path.join(self._output_path, xml_file), "rb") as f:
                output_model.set_data("openvino.xml", f.read())
            output_model.set_data(
                "confidence_threshold", np.array([self.confidence_threshold], dtype=np.float32).tobytes()
            )
            output_model.precision = self._precision
            output_model.optimization_methods = self._optimization_methods
        except Exception as ex:
            raise RuntimeError("Optimization was unsuccessful.") from ex
        output_model.set_data("label_schema.json", label_schema_to_bytes(self._task_environment.label_schema))
        logger.info("Exporting completed")

    def _init_recipe_hparam(self) -> dict:
        configs = super()._init_recipe_hparam()
        configs.data.videos_per_gpu = configs.data.pop("samples_per_gpu", None)  # type: ignore[attr-defined]
        self._recipe_cfg.total_epochs = configs.runner.max_epochs  # type: ignore[attr-defined]
        # FIXME lr_config variables are hard-coded
        if hasattr(configs, "lr_config") and hasattr(configs["lr_config"], "warmup_iters"):
            self._recipe_cfg.lr_config.warmup = "linear"  # type: ignore[attr-defined]
            self._recipe_cfg.lr_config.warmup_by_epoch = True  # type: ignore[attr-defined]
        configs["use_adaptive_interval"] = self._hyperparams.learning_parameters.use_adaptive_interval
        return configs

    def _init_recipe(self):
        logger.info("called _init_recipe()")
        recipe_root = os.path.abspath(os.path.dirname(self.template_file_path))
        recipe = os.path.join(recipe_root, "model.py")
        self._recipe_cfg = Config.fromfile(recipe)
        patch_config(self._recipe_cfg, self.data_pipeline_path, self._output_path, self._task_type)
        set_data_classes(self._recipe_cfg, self._labels, self._task_type)
        logger.info(f"initialized recipe = {recipe}")

    def _init_model_cfg(self):
        model_cfg = Config.fromfile(os.path.join(self._model_dir, "model.py"))
        return model_cfg

    def _add_predictions_to_dataset(self, prediction_results: Iterable, dataset: DatasetEntity):
        """Loop over dataset again to assign predictions. Convert from MM format to OTX format."""
        prediction_results = list(prediction_results)
        video_info: Dict[str, int] = {}
        for dataset_item in dataset:
            video_id = dataset_item.get_metadata()[0].data.video_id
            if video_id not in video_info:
                video_info[video_id] = len(video_info)
        for dataset_item in dataset:
            video_id = dataset_item.get_metadata()[0].data.video_id
            all_results, feature_vector, saliency_map = prediction_results[video_info[video_id]]
            item_labels = []
            label = ScoredLabel(label=self._labels[all_results.argmax()], probability=all_results.max())
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

    def _add_det_predictions_to_dataset(self, prediction_results: Iterable, dataset: DatasetEntity):
        confidence_threshold = 0.05
        self._remove_empty_frames(dataset)
        for dataset_item, (all_results, feature_vector, saliency_map) in zip(dataset, prediction_results):
            shapes = []
            for label_idx, detections in enumerate(all_results):
                for i in range(detections.shape[0]):
                    probability = float(detections[i, 4])
                    coords = detections[i, :4]

                    if probability < confidence_threshold:
                        continue
                    if coords[3] - coords[1] <= 0 or coords[2] - coords[0] <= 0:
                        continue

                    assigned_label = [ScoredLabel(self._labels[label_idx], probability=probability)]
                    shapes.append(
                        Annotation(
                            Rectangle(x1=coords[0], y1=coords[1], x2=coords[2], y2=coords[3]),
                            labels=assigned_label,
                        )
                    )
            dataset_item.append_annotations(shapes)

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
