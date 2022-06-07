# Copyright (C) 2021 Intel Corporation
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
import math
import os
import shutil
import tempfile
import warnings
from subprocess import run  # nosec
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, load_state_dict
from mmcv.utils import Config
from ote_sdk.entities.annotation import Annotation
from ote_sdk.entities.datasets import DatasetEntity
from ote_sdk.entities.id import ID
from ote_sdk.entities.inference_parameters import InferenceParameters, default_progress_callback
from ote_sdk.entities.model import ModelEntity, ModelFormat, ModelOptimizationType, ModelPrecision, OptimizationMethod
from ote_sdk.entities.model_template import TaskType, task_type_to_label_domain
from ote_sdk.entities.resultset import ResultSetEntity
from ote_sdk.entities.scored_label import ScoredLabel
from ote_sdk.entities.shapes.polygon import Point, Polygon
from ote_sdk.entities.shapes.rectangle import Rectangle
from ote_sdk.entities.task_environment import TaskEnvironment
from ote_sdk.entities.tensor import TensorEntity
from ote_sdk.usecases.evaluation.metrics_helper import MetricsHelper
from ote_sdk.usecases.tasks.interfaces.evaluate_interface import IEvaluationTask
from ote_sdk.usecases.tasks.interfaces.export_interface import ExportType, IExportTask
from ote_sdk.usecases.tasks.interfaces.inference_interface import IInferenceTask
from ote_sdk.usecases.tasks.interfaces.unload_interface import IUnload
from ote_sdk.serialization.label_mapper import label_schema_to_bytes

from mmdet.apis import export_model
from detection_tasks.apis.detection.config_utils import patch_config, prepare_for_testing, set_hyperparams
from detection_tasks.apis.detection.configuration import OTEDetectionConfig
from detection_tasks.apis.detection.ote_utils import InferenceProgressCallback
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector
from mmdet.parallel import MMDataCPU
from mmdet.utils.collect_env import collect_env
from mmdet.utils.logger import get_root_logger

logger = get_root_logger()


class OTEDetectionInferenceTask(IInferenceTask, IExportTask, IEvaluationTask, IUnload):

    _task_environment: TaskEnvironment

    def __init__(self, task_environment: TaskEnvironment):
        """"
        Task for inference object detection models using OTEDetection.
        """
        logger.info('Loading OTEDetectionTask')

        print('ENVIRONMENT:')
        for name, val in collect_env().items():
            print(f'{name}: {val}')
        print('pip list:')
        run('pip list', shell=True, check=True)

        self._task_environment = task_environment
        self._task_type = task_environment.model_template.task_type
        self._scratch_space = tempfile.mkdtemp(prefix="ote-det-scratch-")
        logger.info(f'Scratch space created at {self._scratch_space}')

        self._model_name = task_environment.model_template.name
        self._labels = task_environment.get_labels(False)

        template_file_path = task_environment.model_template.model_template_path

        # Get and prepare mmdet config.
        self._base_dir = os.path.abspath(os.path.dirname(template_file_path))
        config_file_path = os.path.join(self._base_dir, "model.py")
        self._config = Config.fromfile(config_file_path)
        patch_config(self._config, self._scratch_space, self._labels, task_type_to_label_domain(self._task_type), random_seed=42)
        set_hyperparams(self._config, self._hyperparams)
        self.confidence_threshold: float = self._hyperparams.postprocessing.confidence_threshold

        # Set default model attributes.
        self._optimization_methods = []
        self._precision = [ModelPrecision.FP16] if self._config.get('fp16', None) else [ModelPrecision.FP32]
        self._optimization_type = ModelOptimizationType.MO

        # Create and initialize PyTorch model.
        logger.info('Loading the model')
        self._model = self._load_model(task_environment.model)

        # Extra control variables.
        self._training_work_dir = None
        self._is_training = False
        self._should_stop = False
        logger.info('Task initialization completed')

    @property
    def _hyperparams(self):
        return self._task_environment.get_hyper_parameters(OTEDetectionConfig)

    def _load_model(self, model: ModelEntity):
        if model is not None:
            # If a model has been trained and saved for the task already, create empty model and load weights here
            buffer = io.BytesIO(model.get_data("weights.pth"))
            model_data = torch.load(buffer, map_location=torch.device('cpu'))

            self.confidence_threshold = model_data.get('confidence_threshold', self.confidence_threshold)
            if model_data.get('anchors'):
                anchors = model_data['anchors']
                self._config.model.bbox_head.anchor_generator.heights = anchors['heights']
                self._config.model.bbox_head.anchor_generator.widths = anchors['widths']

            model = self._create_model(self._config, from_scratch=True)

            try:
                load_state_dict(model, model_data['model'])

                if "load_from" in self._config:
                    self._config.load_from = None

                logger.info(f"Loaded model weights from Task Environment")
                logger.info(f"Model architecture: {self._model_name}")
                for name, weights in model.named_parameters():
                    if(not torch.isfinite(weights).all()):
                        logger.info(f"Invalid weights in: {name}. Recreate model from pre-trained weights")
                        model = self._create_model(self._config, from_scratch=False)
                        return model

            except BaseException as ex:
                raise ValueError("Could not load the saved model. The model file structure is invalid.") \
                    from ex
        else:
            # If there is no trained model yet, create model with pretrained weights as defined in the model config
            # file.
            model = self._create_model(self._config, from_scratch=False)
            logger.info(f"No trained model in project yet. Created new model with '{self._model_name}' "
                        f"architecture and general-purpose pretrained weights.")
        return model


    @staticmethod
    def _create_model(config: Config, from_scratch: bool = False):
        """
        Creates a model, based on the configuration in config

        :param config: mmdetection configuration from which the model has to be built
        :param from_scratch: bool, if True does not load any weights

        :return model: ModelEntity in training mode
        """
        model_cfg = copy.deepcopy(config.model)

        init_from = None if from_scratch else config.get('load_from', None)
        logger.warning(init_from)
        if init_from is not None:
            # No need to initialize backbone separately, if all weights are provided.
            model_cfg.pretrained = None
            logger.warning('build detector')
            model = build_detector(model_cfg)
            # Load all weights.
            logger.warning('load checkpoint')
            load_checkpoint(model, init_from, map_location='cpu')
        else:
            logger.warning('build detector')
            model = build_detector(model_cfg)
        return model


    def _add_predictions_to_dataset(self, prediction_results, dataset, confidence_threshold=0.0):
        """ Loop over dataset again to assign predictions. Convert from MMDetection format to OTE format. """
        for dataset_item, (all_results, feature_vector) in zip(dataset, prediction_results):
            width = dataset_item.width
            height = dataset_item.height

            shapes = []
            if self._task_type == TaskType.DETECTION:
                for label_idx, detections in enumerate(all_results):
                    for i in range(detections.shape[0]):
                        probability = float(detections[i, 4])
                        coords = detections[i, :4].astype(float).copy()
                        coords /= np.array([width, height, width, height], dtype=float)
                        coords = np.clip(coords, 0, 1)

                        if probability < confidence_threshold:
                            continue

                        assigned_label = [ScoredLabel(self._labels[label_idx],
                                                      probability=probability)]
                        if coords[3] - coords[1] <= 0 or coords[2] - coords[0] <= 0:
                            continue

                        shapes.append(Annotation(
                            Rectangle(x1=coords[0], y1=coords[1], x2=coords[2], y2=coords[3]),
                            labels=assigned_label))
            elif self._task_type in {TaskType.INSTANCE_SEGMENTATION, TaskType.ROTATED_DETECTION}:
                for label_idx, (boxes, masks) in enumerate(zip(*all_results)):
                    for mask, probability in zip(masks, boxes[:, 4]):
                        mask = mask.astype(np.uint8)
                        probability = float(probability)
                        if math.isnan(probability) or probability < confidence_threshold:
                            continue
                        contours, hierarchies = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
                        if hierarchies is None:
                            continue
                        for contour, hierarchy in zip(contours, hierarchies[0]):
                            if hierarchy[3] != -1 or len(contour) <= 2:
                                continue
                            if self._task_type == TaskType.INSTANCE_SEGMENTATION:
                                points = [Point(x=point[0][0] / width, y=point[0][1] / height) for point in contour]
                            else:
                                box_points = cv2.boxPoints(cv2.minAreaRect(contour))
                                points = [Point(x=point[0] / width, y=point[1] / height) for point in box_points]
                            labels = [ScoredLabel(self._labels[label_idx], probability=probability)]
                            polygon = Polygon(points=points)
                            if cv2.contourArea(contour) > 0 and polygon.get_area() > 1e-12:
                                shapes.append(Annotation(polygon, labels=labels, id=ID(f"{label_idx:08}")))
            else:
                raise RuntimeError(
                    f"Detection results assignment not implemented for task: {self._task_type}")

            dataset_item.append_annotations(shapes)

            if feature_vector is not None:
                active_score = TensorEntity(name="representation_vector", numpy=feature_vector)
                dataset_item.append_metadata_item(active_score, model=self._task_environment.model)


    def infer(self, dataset: DatasetEntity, inference_parameters: Optional[InferenceParameters] = None) -> DatasetEntity:
        """ Analyzes a dataset using the latest inference model. """

        logger.info('Infer the model on the dataset')
        set_hyperparams(self._config, self._hyperparams)
        # There is no need to have many workers for a couple of images.
        self._config.data.workers_per_gpu = max(min(self._config.data.workers_per_gpu, len(dataset) - 1), 0)

        # If confidence threshold is adaptive then up-to-date value should be stored in the model
        # and should not be changed during inference. Otherwise user-specified value should be taken.
        if not self._hyperparams.postprocessing.result_based_confidence_threshold:
            self.confidence_threshold = self._hyperparams.postprocessing.confidence_threshold

        update_progress_callback = default_progress_callback
        if inference_parameters is not None:
            update_progress_callback = inference_parameters.update_progress

        time_monitor = InferenceProgressCallback(len(dataset), update_progress_callback)

        def pre_hook(module, input):
            time_monitor.on_test_batch_begin(None, None)

        def hook(module, input, output):
            time_monitor.on_test_batch_end(None, None)

        logger.info(f'Confidence threshold {self.confidence_threshold}')
        model = self._model
        with model.register_forward_pre_hook(pre_hook), model.register_forward_hook(hook):
            prediction_results, _ = self._infer_detector(model, self._config, dataset, dump_features=True, eval=False)
        self._add_predictions_to_dataset(prediction_results, dataset, self.confidence_threshold)

        logger.info('Inference completed')
        return dataset


    @staticmethod
    def _infer_detector(model: torch.nn.Module, config: Config, dataset: DatasetEntity, dump_features: bool = False,
                        eval: Optional[bool] = False, metric_name: Optional[str] = 'mAP') -> Tuple[List, float]:
        model.eval()
        test_config = prepare_for_testing(config, dataset)
        mm_val_dataset = build_dataset(test_config.data.test)
        batch_size = 1
        mm_val_dataloader = build_dataloader(mm_val_dataset,
                                             samples_per_gpu=batch_size,
                                             workers_per_gpu=test_config.data.workers_per_gpu,
                                             num_gpus=1,
                                             dist=False,
                                             shuffle=False)
        if torch.cuda.is_available():
            eval_model = MMDataParallel(model.cuda(test_config.gpu_ids[0]),
                                        device_ids=test_config.gpu_ids)
        else:
            eval_model = MMDataCPU(model)

        eval_predictions = []
        feature_vectors = []

        def dump_features_hook(mod, inp, out):
            with torch.no_grad():
                feature_map = out[-1]
                feature_vector = torch.nn.functional.adaptive_avg_pool2d(feature_map, (1, 1))
                assert feature_vector.size(0) == 1
            feature_vectors.append(feature_vector.view(-1).detach().cpu().numpy())

        def dummy_dump_features_hook(mod, inp, out):
            feature_vectors.append(None)

        hook = dump_features_hook if dump_features else dummy_dump_features_hook

        # Use a single gpu for testing. Set in both mm_val_dataloader and eval_model
        with eval_model.module.backbone.register_forward_hook(hook):
            for data in mm_val_dataloader:
                with torch.no_grad():
                    result = eval_model(return_loss=False, rescale=True, **data)
                eval_predictions.extend(result)

        # hard-code way to remove EvalHook args
        for key in [
                'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                'rule', 'dynamic_intervals'
        ]:
            config.evaluation.pop(key, None)

        metric = None
        if eval:
            metric = mm_val_dataset.evaluate(eval_predictions, **config.evaluation)[metric_name]

        assert len(eval_predictions) == len(feature_vectors), f'{len(eval_predictions)} != {len(feature_vectors)}'
        eval_predictions = zip(eval_predictions, feature_vectors)
        return eval_predictions, metric


    def evaluate(self,
                 output_result_set: ResultSetEntity,
                 evaluation_metric: Optional[str] = None):
        """ Computes performance on a resultset """
        logger.info('Evaluating the metric')
        if evaluation_metric is not None:
            logger.warning(f'Requested to use {evaluation_metric} metric, but parameter is ignored. Use F-measure instead.')
        metric = MetricsHelper.compute_f_measure(output_result_set)
        logger.info(f"F-measure after evaluation: {metric.f_measure.value}")
        output_result_set.performance = metric.get_performance()
        logger.info('Evaluation completed')


    @staticmethod
    def _is_docker():
        """
        Checks whether the task runs in docker container

        :return bool: True if task runs in docker
        """
        path = '/proc/self/cgroup'
        is_in_docker = False
        if os.path.isfile(path):
            with open(path) as f:
                is_in_docker = is_in_docker or any('docker' in line for line in f)
        is_in_docker = is_in_docker or os.path.exists('/.dockerenv')
        return is_in_docker

    def unload(self):
        """
        Unload the task
        """
        self._delete_scratch_space()
        if self._is_docker():
            logger.warning(
                "Got unload request. Unloading models. Throwing Segmentation Fault on purpose")
            import ctypes
            ctypes.string_at(0)
        else:
            logger.warning("Got unload request, but not on Docker. Only clearing CUDA cache")
            torch.cuda.empty_cache()
            logger.warning(f"Done unloading. "
                           f"Torch is still occupying {torch.cuda.memory_allocated()} bytes of GPU memory")

    def export(self,
               export_type: ExportType,
               output_model: ModelEntity):
        logger.info('Exporting the model')
        assert export_type == ExportType.OPENVINO
        output_model.model_format = ModelFormat.OPENVINO
        output_model.optimization_type = self._optimization_type
        with tempfile.TemporaryDirectory() as tempdir:
            optimized_model_dir = os.path.join(tempdir, 'export')
            logger.info(f'Optimized model will be temporarily saved to "{optimized_model_dir}"')
            os.makedirs(optimized_model_dir, exist_ok=True)
            try:
                from torch.jit._trace import TracerWarning
                warnings.filterwarnings('ignore', category=TracerWarning)
                if torch.cuda.is_available():
                    model = self._model.cuda(self._config.gpu_ids[0])
                else:
                    model = self._model.cpu()
                pruning_transformation = OptimizationMethod.FILTER_PRUNING in self._optimization_methods
                export_model(model, self._config, tempdir, target='openvino',
                             pruning_transformation=pruning_transformation, precision=self._precision[0].name)
                bin_file = [f for f in os.listdir(tempdir) if f.endswith('.bin')][0]
                xml_file = [f for f in os.listdir(tempdir) if f.endswith('.xml')][0]
                with open(os.path.join(tempdir, bin_file), "rb") as f:
                    output_model.set_data('openvino.bin', f.read())
                with open(os.path.join(tempdir, xml_file), "rb") as f:
                    output_model.set_data('openvino.xml', f.read())
                output_model.set_data('confidence_threshold', np.array([self.confidence_threshold], dtype=np.float32).tobytes())
                output_model.precision = self._precision
                output_model.optimization_methods = self._optimization_methods
            except Exception as ex:
                raise RuntimeError('Optimization was unsuccessful.') from ex
        output_model.set_data("label_schema.json", label_schema_to_bytes(self._task_environment.label_schema))
        logger.info('Exporting completed')

    def _delete_scratch_space(self):
        """
        Remove model checkpoints and mmdet logs
        """
        if os.path.exists(self._scratch_space):
            shutil.rmtree(self._scratch_space, ignore_errors=False)
