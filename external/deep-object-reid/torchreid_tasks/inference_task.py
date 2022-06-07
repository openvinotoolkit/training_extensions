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

import io
import logging
import math
import os
import shutil
import tempfile
from typing import Dict, Optional

import torch

import torchreid
from ote_sdk.configuration import cfg_helper
from ote_sdk.configuration.helper.utils import ids_to_strings
from ote_sdk.entities.datasets import DatasetEntity
from ote_sdk.entities.inference_parameters import InferenceParameters
from ote_sdk.entities.metadata import FloatMetadata, FloatType
from ote_sdk.entities.model import (ModelEntity, ModelFormat, ModelOptimizationType,
                                    ModelPrecision)
from ote_sdk.entities.model import OptimizationMethod
from ote_sdk.entities.result_media import ResultMediaEntity
from ote_sdk.entities.resultset import ResultSetEntity
from ote_sdk.entities.scored_label import ScoredLabel
from ote_sdk.entities.task_environment import TaskEnvironment
from ote_sdk.entities.tensor import TensorEntity
from ote_sdk.entities.train_parameters import default_progress_callback
from ote_sdk.serialization.label_mapper import label_schema_to_bytes
from ote_sdk.usecases.evaluation.metrics_helper import MetricsHelper
from ote_sdk.usecases.tasks.interfaces.evaluate_interface import IEvaluationTask
from ote_sdk.usecases.tasks.interfaces.export_interface import ExportType, IExportTask
from ote_sdk.usecases.tasks.interfaces.inference_interface import IInferenceTask
from ote_sdk.usecases.tasks.interfaces.unload_interface import IUnload
from ote_sdk.utils.labels_utils import get_empty_label
from scripts.default_config import (get_default_config, imagedata_kwargs,
                                    merge_from_files_with_base, model_kwargs)
from torchreid.apis.export import export_ir, export_onnx
from torchreid_tasks.monitors import DefaultMetricsMonitor, StopCallback
from torchreid_tasks.parameters import OTEClassificationParameters
from torchreid_tasks.utils import (active_score_from_probs, force_fp32, get_actmap, get_multiclass_predictions,
                                            get_multilabel_predictions, InferenceProgressCallback,
                                            OTEClassificationDataset, preprocess_features_for_actmap,
                                            sigmoid_numpy, softmax_numpy, get_multihead_class_info,
                                            get_hierarchical_predictions)
from torchreid.metrics.classification import score_extraction
from torchreid.utils import load_pretrained_weights

logger = logging.getLogger(__name__)


class OTEClassificationInferenceTask(IInferenceTask, IEvaluationTask, IExportTask, IUnload):

    task_environment: TaskEnvironment

    def __init__(self, task_environment: TaskEnvironment):
        logger.info("Loading OTEClassificationTask.")
        self._scratch_space = tempfile.mkdtemp(prefix="ote-cls-scratch-")
        logger.info(f"Scratch space created at {self._scratch_space}")

        self._task_environment = task_environment
        if len(task_environment.get_labels(False)) == 1:
            self._labels = task_environment.get_labels(include_empty=True)
        else:
            self._labels = task_environment.get_labels(include_empty=False)
        self._empty_label = get_empty_label(task_environment.label_schema)
        self._multilabel = len(task_environment.label_schema.get_groups(False)) > 1 and \
                len(task_environment.label_schema.get_groups(False)) == \
                len(task_environment.get_labels(include_empty=False))

        self._multihead_class_info = {}
        self._hierarchical = False
        if not self._multilabel and len(task_environment.label_schema.get_groups(False)) > 1:
            self._hierarchical = True
            self._multihead_class_info = get_multihead_class_info(task_environment.label_schema)

        template_file_path = task_environment.model_template.model_template_path

        self._base_dir = os.path.abspath(os.path.dirname(template_file_path))

        self._cfg = get_default_config()
        self._patch_config(self._base_dir)

        if self._multilabel:
            assert self._cfg.model.type == 'multilabel', task_environment.model_template.model_template_path + \
                ' model template does not support multilabel classification'
        elif self._hierarchical:
            assert self._cfg.model.type == 'multihead', task_environment.model_template.model_template_path + \
                ' model template does not support hierarchical classification'
        else:
            assert self._cfg.model.type == 'classification', task_environment.model_template.model_template_path + \
                ' model template does not support multiclass classification'

        self.device = torch.device("cuda:0") if torch.cuda.device_count() else torch.device("cpu")
        self._model = self._load_model(task_environment.model, device=self.device)

        self.stop_callback = StopCallback()
        self.metrics_monitor = DefaultMetricsMonitor()

        # Set default model attributes.
        self._optimization_methods = []
        self._precision = [ModelPrecision.FP32]
        self._optimization_type = ModelOptimizationType.MO

    @property
    def _hyperparams(self):
        return self._task_environment.get_hyper_parameters(OTEClassificationParameters)

    def _load_model(self, model: ModelEntity, device: torch.device, pretrained_dict: Optional[Dict] = None):
        if model is not None:
            # If a model has been trained and saved for the task already, create empty model and load weights here
            if pretrained_dict is None:
                buffer = io.BytesIO(model.get_data("weights.pth"))
                model_data = torch.load(buffer, map_location=torch.device('cpu'))
            else:
                model_data = pretrained_dict

            model = self._create_model(self._cfg, from_scratch=True)

            try:
                load_pretrained_weights(model, pretrained_dict=model_data)
                logger.info("Loaded model weights from Task Environment")
            except BaseException as ex:
                raise ValueError("Could not load the saved model. The model file structure is invalid.") \
                    from ex
        else:
            # If there is no trained model yet, create model with pretrained weights as defined in the model config
            # file.
            model = self._create_model(self._cfg, from_scratch=False)
            logger.info("No trained model in project yet. Created new model with general-purpose pretrained weights.")
        return model.to(device)

    def _create_model(self, config, from_scratch: bool = False):
        """
        Creates a model, based on the configuration in config
        :param config: deep-object-reid configuration from which the model has to be built
        :param from_scratch: bool, if True does not load any weights
        :return model: Model in training mode
        """
        num_train_classes = len(self._labels)
        model = torchreid.models.build_model(**model_kwargs(config, num_train_classes))
        if self._cfg.model.load_weights and not from_scratch:
            load_pretrained_weights(model, self._cfg.model.load_weights)
        return model

    def _patch_config(self, base_dir: str):
        self._cfg = get_default_config()
        if self._multilabel:
            config_file_path = os.path.join(base_dir, 'main_model_multilabel.yaml')
        elif self._hierarchical:
            config_file_path = os.path.join(base_dir, 'main_model_multihead.yaml')
        else:
            config_file_path = os.path.join(base_dir, 'main_model.yaml')
        merge_from_files_with_base(self._cfg, config_file_path)
        self._cfg.use_gpu = torch.cuda.device_count() > 0
        self.num_devices = 1 if self._cfg.use_gpu else 0
        if not self._cfg.use_gpu:
            self._cfg.train.mix_precision = False

        self._cfg.custom_datasets.types = ['external_classification_wrapper', 'external_classification_wrapper']
        self._cfg.custom_datasets.roots = ['']*2
        self._cfg.data.save_dir = self._scratch_space

        self._cfg.test.test_before_train = False
        self.num_classes = len(self._labels)

        for i, conf in enumerate(self._cfg.mutual_learning.aux_configs):
            if str(base_dir) not in conf:
                self._cfg.mutual_learning.aux_configs[i] = os.path.join(base_dir, conf)

        self._cfg.train.lr = self._hyperparams.learning_parameters.learning_rate
        self._cfg.train.batch_size = self._hyperparams.learning_parameters.batch_size
        self._cfg.test.batch_size = max(1, self._hyperparams.learning_parameters.batch_size // 2)
        self._cfg.train.max_epoch = self._hyperparams.learning_parameters.max_num_epochs
        self._cfg.lr_finder.enable = self._hyperparams.learning_parameters.enable_lr_finder
        self._cfg.train.early_stopping = self._hyperparams.learning_parameters.enable_early_stopping

    def infer(self, dataset: DatasetEntity,
              inference_parameters: Optional[InferenceParameters] = None) -> DatasetEntity:
        """
        Perform inference on the given dataset.

        :param dataset: Dataset entity to analyse
        :param inference_parameters: Additional parameters for inference.
            For example, when results are generated for evaluation purposes, Saliency maps can be turned off.
        :return: Dataset that also includes the classification results
        """
        if len(dataset) == 0:
            logger.warning("Empty dataset has been passed for the inference.")
            return dataset

        if inference_parameters is not None:
            update_progress_callback = inference_parameters.update_progress
        else:
            update_progress_callback = default_progress_callback

        self._cfg.test.batch_size = max(1, self._hyperparams.learning_parameters.batch_size // 2)
        self._cfg.data.workers = max(min(self._cfg.data.workers, len(dataset) - 1), 0)

        time_monitor = InferenceProgressCallback(math.ceil(len(dataset) / self._cfg.test.batch_size),
                                                 update_progress_callback)

        data = OTEClassificationDataset(dataset, self._labels, self._multilabel,
                                        self._hierarchical, self._multihead_class_info,
                                        keep_empty_label=self._empty_label in self._labels)
        self._cfg.custom_datasets.roots = [data, data]
        datamanager = torchreid.data.ImageDataManager(**imagedata_kwargs(self._cfg))
        with force_fp32(self._model):
            self._model.eval()
            self._model.to(self.device)
            dump_features = not inference_parameters.is_evaluation
            inference_results, _ = score_extraction(datamanager.test_loader,
                                                    self._model, self._cfg.use_gpu,
                                                    perf_monitor=time_monitor,
                                                    feature_dump_mode='all' if dump_features else 'vecs')
        if dump_features:
            scores, features, feature_vecs = inference_results
            features = preprocess_features_for_actmap(features)
        else:
            scores, feature_vecs = inference_results

        if self._multilabel:
            scores = sigmoid_numpy(scores)

        for i in range(scores.shape[0]):
            dataset_item = dataset[i]

            if self._multilabel:
                item_labels = get_multilabel_predictions(scores[i], self._labels, activate=False)
            elif self._hierarchical:
                item_labels = get_hierarchical_predictions(scores[i], self._labels, self._task_environment.label_schema,
                                                           self._multihead_class_info, activate=True)
            else:
                scores[i] = softmax_numpy(scores[i])
                item_labels = get_multiclass_predictions(scores[i], self._labels, activate=False)

            if not item_labels:
                item_labels = [ScoredLabel(self._empty_label, probability=1.)]

            dataset_item.append_labels(item_labels)
            active_score = active_score_from_probs(scores[i])
            active_score_media = FloatMetadata(name="active_score", value=active_score,
                                               float_type=FloatType.ACTIVE_SCORE)
            dataset_item.append_metadata_item(active_score_media, model=self._task_environment.model)
            feature_vec_media = TensorEntity(name="representation_vector", numpy=feature_vecs[i])
            dataset_item.append_metadata_item(feature_vec_media, model=self._task_environment.model)

            if dump_features:
                actmap = get_actmap(features[i], (dataset_item.width, dataset_item.height))
                saliency_media = ResultMediaEntity(name="saliency_map", type="Saliency map",
                                                   annotation_scene=dataset_item.annotation_scene,
                                                   numpy=actmap, roi=dataset_item.roi, label = item_labels[0].label)
                dataset_item.append_metadata_item(saliency_media, model=self._task_environment.model)

        return dataset

    def evaluate(
        self, output_resultset: ResultSetEntity, evaluation_metric: Optional[str] = None
    ):
        performance = MetricsHelper.compute_accuracy(output_resultset).get_performance()
        logger.info(f"Computes performance of {performance}")
        output_resultset.performance = performance

    def export(self, export_type: ExportType, output_model: ModelEntity):
        assert export_type == ExportType.OPENVINO
        output_model.model_format = ModelFormat.OPENVINO
        output_model.optimization_type = self._optimization_type

        with tempfile.TemporaryDirectory() as tempdir:
            optimized_model_dir = os.path.join(tempdir, "dor")
            logger.info(f'Optimized model will be temporarily saved to "{optimized_model_dir}"')
            os.makedirs(optimized_model_dir, exist_ok=True)
            try:
                onnx_model_path = os.path.join(optimized_model_dir, 'model.onnx')
                with force_fp32(self._model):
                    self._model.old_forward = self._model.forward
                    self._model.forward = lambda x: self._model.old_forward(x, return_all=True, apply_scale=True)
                    export_onnx(self._model.eval(), self._cfg, onnx_model_path,
                                opset=self._cfg.model.export_onnx_opset, output_names=['logits', 'features', 'vector'])
                    self._model.forward = self._model.old_forward
                    del self._model.old_forward
                pruning_transformation = OptimizationMethod.FILTER_PRUNING in self._optimization_methods
                export_ir(onnx_model_path, self._cfg.data.norm_mean, self._cfg.data.norm_std,
                          optimized_model_dir=optimized_model_dir, pruning_transformation=pruning_transformation)

                bin_file = [f for f in os.listdir(optimized_model_dir) if f.endswith('.bin')][0]
                xml_file = [f for f in os.listdir(optimized_model_dir) if f.endswith('.xml')][0]
                with open(os.path.join(optimized_model_dir, bin_file), "rb") as f:
                    output_model.set_data("openvino.bin", f.read())
                with open(os.path.join(optimized_model_dir, xml_file), "rb") as f:
                    output_model.set_data("openvino.xml", f.read())
                output_model.precision = self._precision
                output_model.optimization_methods = self._optimization_methods
            except Exception as ex:
                raise RuntimeError('Optimization was unsuccessful.') from ex

        output_model.set_data("label_schema.json", label_schema_to_bytes(self._task_environment.label_schema))
        logger.info('Exporting completed.')

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

    def _delete_scratch_space(self):
        """
        Remove model checkpoints and logs
        """
        if os.path.exists(self._scratch_space):
            shutil.rmtree(self._scratch_space, ignore_errors=False)

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

    def _save_model(self, output_model: ModelEntity, state_dict: Optional[Dict] = None):
        """
        Save model
        """
        buffer = io.BytesIO()
        hyperparams = self._task_environment.get_hyper_parameters(OTEClassificationParameters)
        hyperparams_str = ids_to_strings(cfg_helper.convert(hyperparams, dict, enum_to_str=True))
        modelinfo = {
            'model': self._model.state_dict(),
            'config': hyperparams_str,
            'VERSION': 1
        }

        if state_dict is not None:
            modelinfo.update(state_dict)

        torch.save(modelinfo, buffer)
        output_model.set_data('weights.pth', buffer.getvalue())
        output_model.set_data('label_schema.json', label_schema_to_bytes(self._task_environment.label_schema))
