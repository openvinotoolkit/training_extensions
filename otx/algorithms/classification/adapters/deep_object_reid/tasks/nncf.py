"""NNCF Task for OTX Classification."""

# Copyright (C) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import io
import logging
import math
import os
import shutil
import tempfile
from typing import Any, Dict, List, Optional

import torch
import torchreid
from torchreid.apis.export import export_ir, export_onnx
from torchreid.apis.training import run_training
from torchreid.integration.nncf.compression import (
    check_nncf_is_enabled,
    is_nncf_state,
    wrap_nncf_model,
)
from torchreid.integration.nncf.compression_script_utils import (
    calculate_lr_for_nncf_training,
    patch_config,
)
from torchreid.metrics.classification import score_extraction
from torchreid.ops import DataParallel
from torchreid.utils import load_pretrained_weights, set_model_attr, set_random_seed

from otx.algorithms.classification.adapters.deep_object_reid.scripts import (
    get_default_config,
    imagedata_kwargs,
    lr_scheduler_kwargs,
    merge_from_files_with_base,
    model_kwargs,
    optimizer_kwargs,
)
from otx.algorithms.classification.adapters.deep_object_reid.utils.monitors import (
    DefaultMetricsMonitor,
    StopCallback,
)
from otx.algorithms.classification.adapters.deep_object_reid.utils.parameters import (
    ClassificationParameters,
)
from otx.algorithms.classification.adapters.deep_object_reid.utils.utils import (
    ClassificationDataset,
    InferenceProgressCallback,
    OptimizationProgressCallback,
    active_score_from_probs,
    force_fp32,
    get_hierarchical_predictions,
    get_multiclass_predictions,
    get_multihead_class_info,
    get_multilabel_predictions,
    sigmoid_numpy,
    softmax_numpy,
)
from otx.api.configuration import cfg_helper
from otx.api.configuration.helper.utils import ids_to_strings
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.inference_parameters import InferenceParameters
from otx.api.entities.metadata import FloatMetadata, FloatType
from otx.api.entities.model import (
    ModelEntity,
    ModelFormat,
    ModelOptimizationType,
    ModelPrecision,
    OptimizationMethod,
)
from otx.api.entities.model_template import parse_model_template
from otx.api.entities.optimization_parameters import OptimizationParameters
from otx.api.entities.result_media import ResultMediaEntity
from otx.api.entities.resultset import ResultSetEntity
from otx.api.entities.scored_label import ScoredLabel
from otx.api.entities.subset import Subset
from otx.api.entities.task_environment import TaskEnvironment
from otx.api.entities.tensor import TensorEntity
from otx.api.entities.train_parameters import default_progress_callback
from otx.api.serialization.label_mapper import label_schema_to_bytes
from otx.api.usecases.evaluation.metrics_helper import MetricsHelper
from otx.api.usecases.tasks.interfaces.evaluate_interface import IEvaluationTask
from otx.api.usecases.tasks.interfaces.export_interface import ExportType, IExportTask
from otx.api.usecases.tasks.interfaces.inference_interface import IInferenceTask
from otx.api.usecases.tasks.interfaces.optimization_interface import (
    IOptimizationTask,
    OptimizationType,
)
from otx.api.usecases.tasks.interfaces.unload_interface import IUnload
from otx.api.utils.argument_checks import (
    DatasetParamTypeCheck,
    check_input_parameters_type,
)
from otx.api.utils.labels_utils import get_empty_label
from otx.api.utils.vis_utils import get_actmap

logger = logging.getLogger(__name__)


class ClassificationInferenceTask(
    IInferenceTask, IEvaluationTask, IExportTask, IUnload
):  # pylint: disable=too-many-instance-attributes
    """Inference task running through deep-object-reid."""

    task_environment: TaskEnvironment

    @check_input_parameters_type()
    def __init__(self, task_environment: TaskEnvironment):
        logger.info("Loading ClassificationTask.")
        self._scratch_space = tempfile.mkdtemp(prefix="otx-cls-scratch-")
        logger.info(f"Scratch space created at {self._scratch_space}")

        self._task_environment = task_environment
        if len(task_environment.get_labels(False)) == 1:
            self._labels = task_environment.get_labels(include_empty=True)
        else:
            self._labels = task_environment.get_labels(include_empty=False)
        self._empty_label = get_empty_label(task_environment.label_schema)
        self._multilabel = len(task_environment.label_schema.get_groups(False)) > 1 and len(
            task_environment.label_schema.get_groups(False)
        ) == len(task_environment.get_labels(include_empty=False))

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
            assert self._cfg.model.type == "multilabel", (
                task_environment.model_template.model_template_path
                + " model template does not support multilabel classification"
            )
        elif self._hierarchical:
            assert self._cfg.model.type == "multihead", (
                task_environment.model_template.model_template_path
                + " model template does not support hierarchical classification"
            )
        else:
            assert self._cfg.model.type == "classification", (
                task_environment.model_template.model_template_path
                + " model template does not support multiclass classification"
            )

        self.device = torch.device("cuda:0") if torch.cuda.device_count() else torch.device("cpu")
        self._model = self._load_model(task_environment.model, device=self.device)
        self._model.to(self.device)

        self.stop_callback = StopCallback()
        self.metrics_monitor = DefaultMetricsMonitor()

        # Set default model attributes.
        self._optimization_methods = []  # type: List[Any]
        self._precision = [ModelPrecision.FP32]
        self._optimization_type = ModelOptimizationType.MO

    @property
    def _hyperparams(self):
        return self._task_environment.get_hyper_parameters(ClassificationParameters)

    def _load_model(
        self, model: Optional[ModelEntity], device: torch.device, pretrained_dict: Optional[Dict] = None
    ):  # pylint: disable=unused-argument
        if model is not None:
            # If a model has been trained and saved for the task already, create empty model and load weights here
            if pretrained_dict is None:
                buffer = io.BytesIO(model.get_data("weights.pth"))
                model_data = torch.load(buffer, map_location=torch.device("cpu"))
            else:
                model_data = pretrained_dict

            model = self._create_model(self._cfg, from_scratch=True)

            try:
                load_pretrained_weights(model, pretrained_dict=model_data)
                logger.info("Loaded model weights from Task Environment")
            except BaseException as ex:
                raise ValueError("Could not load the saved model. The model file structure is invalid.") from ex
        else:
            # If there is no trained model yet, create model with pretrained weights as defined in the model config
            # file.
            model = self._create_model(self._cfg, from_scratch=False)
            logger.info("No trained model in project yet. Created new model with general-purpose pretrained weights.")

        return model

    def _create_model(self, config, from_scratch: bool = False):
        """Creates a model, based on the configuration in config.

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
            config_file_path = os.path.join(base_dir, "main_model_multilabel.yaml")
        elif self._hierarchical:
            config_file_path = os.path.join(base_dir, "main_model_multihead.yaml")
        else:
            config_file_path = os.path.join(base_dir, "main_model.yaml")
        merge_from_files_with_base(self._cfg, config_file_path)
        self._cfg.use_gpu = torch.cuda.device_count() > 0
        self.num_devices = 1 if self._cfg.use_gpu else 0
        if not self._cfg.use_gpu:
            self._cfg.train.mix_precision = False

        self._cfg.custom_datasets.types = ["external_classification_wrapper", "external_classification_wrapper"]
        self._cfg.custom_datasets.roots = [""] * 2
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

    # pylint: disable=too-many-locals, too-many-branches
    @check_input_parameters_type({"dataset": DatasetParamTypeCheck})
    def infer(
        self, dataset: DatasetEntity, inference_parameters: Optional[InferenceParameters] = None
    ) -> DatasetEntity:
        """Perform inference on the given dataset.

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

        time_monitor = InferenceProgressCallback(
            math.ceil(len(dataset) / self._cfg.test.batch_size), update_progress_callback
        )

        data = ClassificationDataset(
            dataset,
            self._labels,
            self._multilabel,
            self._hierarchical,
            self._multihead_class_info,
            keep_empty_label=self._empty_label in self._labels,
        )
        self._cfg.custom_datasets.roots = [data, data]
        datamanager = torchreid.data.ImageDataManager(**imagedata_kwargs(self._cfg))
        with force_fp32(self._model):
            self._model.eval()
            self._model.to(self.device)
            if inference_parameters is not None:
                dump_features = not inference_parameters.is_evaluation
            inference_results, _ = score_extraction(
                datamanager.test_loader,
                self._model,
                self._cfg.use_gpu,
                perf_monitor=time_monitor,
                feature_dump_mode="all" if dump_features else "vecs",
            )
        if dump_features:
            scores, saliency_maps, feature_vecs = inference_results
        else:
            scores, feature_vecs = inference_results

        if self._multilabel:
            scores = sigmoid_numpy(scores)

        for i in range(scores.shape[0]):
            dataset_item = dataset[i]

            if self._multilabel:
                item_labels = get_multilabel_predictions(scores[i], self._labels, activate=False)
            elif self._hierarchical:
                item_labels = get_hierarchical_predictions(
                    scores[i],
                    self._labels,
                    self._task_environment.label_schema,
                    self._multihead_class_info,
                    activate=True,
                )
            else:
                scores[i] = softmax_numpy(scores[i])
                item_labels = get_multiclass_predictions(scores[i], self._labels, activate=False)

            if not item_labels:
                if self._empty_label is not None:
                    item_labels = [ScoredLabel(self._empty_label, probability=1.0)]

            dataset_item.append_labels(item_labels)
            active_score = active_score_from_probs(scores[i])
            active_score_media = FloatMetadata(
                name="active_score", value=active_score, float_type=FloatType.ACTIVE_SCORE
            )
            dataset_item.append_metadata_item(active_score_media, model=self._task_environment.model)
            feature_vec_media = TensorEntity(name="representation_vector", numpy=feature_vecs[i].reshape(-1))
            dataset_item.append_metadata_item(feature_vec_media, model=self._task_environment.model)

            if dump_features:
                actmap = get_actmap(saliency_maps[i], (dataset_item.width, dataset_item.height))
                saliency_media = ResultMediaEntity(
                    name="Saliency Map",
                    type="saliency_map",
                    annotation_scene=dataset_item.annotation_scene,
                    numpy=actmap,
                    roi=dataset_item.roi,
                    label=item_labels[0].label,
                )
                dataset_item.append_metadata_item(saliency_media, model=self._task_environment.model)

        return dataset

    @check_input_parameters_type()
    def evaluate(self, output_resultset: ResultSetEntity, evaluation_metric: Optional[str] = None):
        """Evaluate."""
        performance = MetricsHelper.compute_accuracy(output_resultset).get_performance()
        logger.info(f"Computes performance of {performance}")
        output_resultset.performance = performance

    @check_input_parameters_type()
    def export(self, export_type: ExportType, output_model: ModelEntity):
        """Export."""
        assert export_type == ExportType.OPENVINO
        output_model.model_format = ModelFormat.OPENVINO
        output_model.optimization_type = self._optimization_type

        with tempfile.TemporaryDirectory() as tempdir:
            optimized_model_dir = os.path.join(tempdir, "deep_object_reid")
            logger.info(f'Optimized model will be temporarily saved to "{optimized_model_dir}"')
            os.makedirs(optimized_model_dir, exist_ok=True)
            try:
                onnx_model_path = os.path.join(optimized_model_dir, "model.onnx")
                with force_fp32(self._model):
                    self._model.old_forward = self._model.forward
                    self._model.forward = lambda x: self._model.old_forward(x, return_all=True, apply_scale=True)
                    export_onnx(
                        self._model.eval(),
                        self._cfg,
                        onnx_model_path,
                        opset=self._cfg.model.export_onnx_opset,
                        output_names=["logits", "saliency_map", "feature_vector"],
                    )
                    self._model.forward = self._model.old_forward
                    del self._model.old_forward
                pruning_transformation = OptimizationMethod.FILTER_PRUNING in self._optimization_methods
                export_ir(
                    onnx_model_path,
                    self._cfg.data.norm_mean,
                    self._cfg.data.norm_std,
                    optimized_model_dir=optimized_model_dir,
                    pruning_transformation=pruning_transformation,
                )

                bin_file = [f for f in os.listdir(optimized_model_dir) if f.endswith(".bin")][0]
                xml_file = [f for f in os.listdir(optimized_model_dir) if f.endswith(".xml")][0]
                with open(os.path.join(optimized_model_dir, bin_file), "rb") as f:
                    output_model.set_data("openvino.bin", f.read())
                with open(os.path.join(optimized_model_dir, xml_file), "rb") as f:
                    output_model.set_data("openvino.xml", f.read())
                output_model.precision = self._precision
                output_model.optimization_methods = self._optimization_methods
            except Exception as ex:
                raise RuntimeError("Optimization was unsuccessful.") from ex

        output_model.set_data("label_schema.json", label_schema_to_bytes(self._task_environment.label_schema))
        logger.info("Exporting completed.")

    @staticmethod
    def _is_docker():
        """Checks whether the task runs in docker container.

        :return bool: True if task runs in docker
        """
        path = "/proc/self/cgroup"
        is_in_docker = False
        if os.path.isfile(path):
            with open(path, "rb") as f:
                is_in_docker = is_in_docker or any("docker" in line for line in f)
        is_in_docker = is_in_docker or os.path.exists("/.dockerenv")
        return is_in_docker

    def _delete_scratch_space(self):
        """Remove model checkpoints and logs."""
        if os.path.exists(self._scratch_space):
            shutil.rmtree(self._scratch_space, ignore_errors=False)

    def unload(self):
        """Unload the task."""
        self._delete_scratch_space()
        if self._is_docker():
            logger.warning("Got unload request. Unloading models. Throwing Segmentation Fault on purpose")
            import ctypes

            ctypes.string_at(0)
        else:
            logger.warning("Got unload request, but not on Docker. Only clearing CUDA cache")
            torch.cuda.empty_cache()
            logger.warning(
                f"Done unloading. " f"Torch is still occupying {torch.cuda.memory_allocated()} bytes of GPU memory"
            )

    def _save_model(self, output_model: ModelEntity, state_dict: Optional[Dict] = None):
        """Save model."""
        buffer = io.BytesIO()
        hyperparams = self._task_environment.get_hyper_parameters(ClassificationParameters)
        hyperparams_str = ids_to_strings(cfg_helper.convert(hyperparams, dict, enum_to_str=True))
        modelinfo = {"model": self._model.state_dict(), "config": hyperparams_str, "VERSION": 1}

        if state_dict is not None:
            modelinfo.update(state_dict)

        torch.save(modelinfo, buffer)
        output_model.set_data("weights.pth", buffer.getvalue())
        output_model.set_data("label_schema.json", label_schema_to_bytes(self._task_environment.label_schema))


class ClassificationNNCFTask(
    ClassificationInferenceTask, IOptimizationTask
):  # pylint: disable=too-many-instance-attributes
    """Task for compressing classification models using NNCF."""

    def __init__(self, task_environment: TaskEnvironment):
        curr_model_path = task_environment.model_template.model_template_path
        base_model_path = os.path.join(
            os.path.dirname(os.path.abspath(curr_model_path)),
            task_environment.model_template.base_model_path,
        )
        if os.path.isfile(base_model_path):
            logger.info(f"Base model for NNCF: {base_model_path}")
            # Redirect to base model
            task_environment.model_template = parse_model_template(base_model_path)
        logger.info("Loading ClassificationNNCFTask.")
        super().__init__(task_environment)

        check_nncf_is_enabled()

        # Set hyperparameters
        self._nncf_preset = None
        self._max_acc_drop = None
        self._set_attributes_by_hyperparams()

        # Patch the config
        if not self._cfg.nncf.nncf_config_path:
            self._cfg.nncf.nncf_config_path = os.path.join(self._base_dir, "compression_config.json")
        self._cfg = patch_config(self._cfg, self._nncf_preset, self._max_acc_drop)

        self._compression_ctrl = None
        self._nncf_metainfo = None

        # Load NNCF model.
        if task_environment.model is not None:
            if task_environment.model.optimization_type == ModelOptimizationType.NNCF:
                logger.info("Loading the NNCF model")
                self._compression_ctrl, self._model, self._nncf_metainfo = self._load_nncf_model(task_environment.model)

        # Set default model attributes.
        self._optimization_type = ModelOptimizationType.NNCF
        logger.info("ClassificationNNCFTask initialization completed")
        set_model_attr(self._model, "mix_precision", self._cfg.train.mix_precision)

    @property
    def _initial_lr(self):
        return getattr(self, "__initial_lr")

    @_initial_lr.setter
    def _initial_lr(self, value):
        setattr(self, "__initial_lr", value)

    def _set_attributes_by_hyperparams(self):
        logger.info("Hyperparameters: ")
        logger.info(
            f"maximal_accuracy_degradation = " f"{self._hyperparams.nncf_optimization.maximal_accuracy_degradation}"
        )
        logger.info(f"enable_quantization = {self._hyperparams.nncf_optimization.enable_quantization}")
        logger.info(f"enable_pruning = {self._hyperparams.nncf_optimization.enable_pruning}")
        self._max_acc_drop = self._hyperparams.nncf_optimization.maximal_accuracy_degradation / 100.0
        quantization = self._hyperparams.nncf_optimization.enable_quantization
        pruning = self._hyperparams.nncf_optimization.enable_pruning
        if quantization and pruning:
            self._nncf_preset = "nncf_quantization_pruning"
            self._optimization_methods = [OptimizationMethod.QUANTIZATION, OptimizationMethod.FILTER_PRUNING]
            self._precision = [ModelPrecision.INT8]
            return
        if quantization and not pruning:
            self._nncf_preset = "nncf_quantization"
            self._optimization_methods = [OptimizationMethod.QUANTIZATION]
            self._precision = [ModelPrecision.INT8]
            return
        if not quantization and pruning:
            self._nncf_preset = "nncf_pruning"
            self._optimization_methods = [OptimizationMethod.FILTER_PRUNING]
            self._precision = [ModelPrecision.FP32]
            return
        raise RuntimeError("Not selected optimization algorithm")

    def _load_model(self, model: Optional[ModelEntity], device: torch.device, pretrained_dict: Optional[Dict] = None):
        if model is None:
            raise ValueError("No trained model in the project. NNCF require pretrained weights to compress the model")

        if model.optimization_type == ModelOptimizationType.NNCF:
            logger.info("Skip loading the original model")
            return None

        model_data = pretrained_dict if pretrained_dict else self._load_model_data(model, "weights.pth")
        if is_nncf_state(model_data):
            raise ValueError("Model optimization type is not consistent with the model checkpoint.")

        self._initial_lr = model_data.get("initial_lr")

        return super()._load_model(model, device, pretrained_dict=model_data)

    def _load_nncf_model(self, model: Optional[ModelEntity]):
        if model is None:
            raise ValueError("No NNCF trained model in project.")

        model_data = self._load_model_data(model, "weights.pth")
        if not is_nncf_state(model_data):
            raise ValueError("Model optimization type is not consistent with the NNCF model checkpoint.")
        model = self._create_model(self._cfg, from_scratch=True)

        compression_ctrl, model, nncf_metainfo = wrap_nncf_model(model, self._cfg, checkpoint_dict=model_data)
        logger.info("Loaded NNCF model weights from Task Environment.")
        return compression_ctrl, model, nncf_metainfo

    def _load_aux_models_data(self, model: Optional[ModelEntity]):
        aux_models_data = []
        num_aux_models = len(self._cfg.mutual_learning.aux_configs)
        if model is None:
            raise TypeError("Model is NoneType.")
        for idx in range(num_aux_models):
            data_name = f"aux_model_{idx + 1}.pth"
            if data_name not in model.model_adapters:
                return []
            model_data = self._load_model_data(model, data_name)
            aux_models_data.append(model_data)
        return aux_models_data

    @check_input_parameters_type({"dataset": DatasetParamTypeCheck})
    def optimize(
        self,
        optimization_type: OptimizationType,
        dataset: DatasetEntity,
        output_model: ModelEntity,
        optimization_parameters: Optional[OptimizationParameters] = None,
    ):  # pylint: disable=too-many-locals
        """Optimize a model on a dataset."""
        if optimization_type is not OptimizationType.NNCF:
            raise RuntimeError("NNCF is the only supported optimization")
        if self._compression_ctrl:
            raise RuntimeError("The model is already optimized. NNCF requires the original model for optimization.")
        if self._cfg.lr_finder.enable:
            raise RuntimeError("LR finder could not be used together with NNCF compression")

        aux_pretrained_dicts = self._load_aux_models_data(self._task_environment.model)
        if len(aux_pretrained_dicts) == 0:
            self._cfg.mutual_learning.aux_configs = []
            logger.warning("WARNING: No pretrained weights are loaded for aux model.")
        num_aux_models = len(self._cfg.mutual_learning.aux_configs)

        if optimization_parameters is not None:
            update_progress_callback = optimization_parameters.update_progress
        else:
            update_progress_callback = default_progress_callback

        num_epoch = self._cfg.nncf_config["accuracy_aware_training"]["params"]["maximal_total_epochs"]
        train_subset = dataset.get_subset(Subset.TRAINING)
        time_monitor = OptimizationProgressCallback(
            update_progress_callback,
            num_epoch=num_epoch,
            num_train_steps=max(1, math.floor(len(train_subset) / self._cfg.train.batch_size)),
            num_val_steps=0,
            num_test_steps=0,
            loading_stage_progress_percentage=5,
            initialization_stage_progress_percentage=5,
        )

        self.metrics_monitor = DefaultMetricsMonitor()
        self.stop_callback.reset()

        set_random_seed(self._cfg.train.seed)
        val_subset = dataset.get_subset(Subset.VALIDATION)
        self._cfg.custom_datasets.roots = [
            ClassificationDataset(
                train_subset,
                self._labels,
                self._multilabel,
                self._hierarchical,
                self._multihead_class_info,
                keep_empty_label=self._empty_label in self._labels,
            ),
            ClassificationDataset(
                val_subset,
                self._labels,
                self._multilabel,
                self._hierarchical,
                self._multihead_class_info,
                keep_empty_label=self._empty_label in self._labels,
            ),
        ]
        datamanager = torchreid.data.ImageDataManager(**imagedata_kwargs(self._cfg))

        self._compression_ctrl, self._model, self._nncf_metainfo = wrap_nncf_model(
            self._model, self._cfg, multihead_info=self._multihead_class_info, datamanager_for_init=datamanager
        )

        time_monitor.on_initialization_end()

        self._cfg.train.lr = calculate_lr_for_nncf_training(self._cfg, self._initial_lr, False)

        train_model = self._model
        if self._cfg.use_gpu:
            main_device_ids = list(range(self.num_devices))
            extra_device_ids = [main_device_ids for _ in range(num_aux_models)]
            train_model = DataParallel(train_model, device_ids=main_device_ids, output_device=0).cuda(
                main_device_ids[0]
            )
        else:
            extra_device_ids = [None for _ in range(num_aux_models)]  # type: ignore

        optimizer = torchreid.optim.build_optimizer(train_model, **optimizer_kwargs(self._cfg))

        scheduler = torchreid.optim.build_lr_scheduler(
            optimizer, num_iter=datamanager.num_iter, **lr_scheduler_kwargs(self._cfg)
        )

        logger.info("Start training")
        time_monitor.on_train_begin()
        run_training(
            self._cfg,
            datamanager,
            train_model,
            optimizer,
            scheduler,
            extra_device_ids,
            self._cfg.train.lr,
            should_freeze_aux_models=True,
            aux_pretrained_dicts=aux_pretrained_dicts,
            tb_writer=self.metrics_monitor,
            perf_monitor=time_monitor,
            stop_callback=self.stop_callback,
            nncf_metainfo=self._nncf_metainfo,
            compression_ctrl=self._compression_ctrl,
        )
        time_monitor.on_train_end()

        self.metrics_monitor.close()
        if self.stop_callback.check_stop():
            logger.info("Training cancelled.")
            return

        logger.info("Training completed")

        self.save_model(output_model)

        output_model.model_format = ModelFormat.BASE_FRAMEWORK
        output_model.optimization_type = self._optimization_type
        output_model.optimization_methods = self._optimization_methods
        output_model.precision = self._precision

    @check_input_parameters_type()
    def save_model(self, output_model: ModelEntity):
        """Save model."""
        state_dict = None
        if self._compression_ctrl is not None:
            state_dict = {
                "compression_state": self._compression_ctrl.get_compression_state(),
                "nncf_metainfo": self._nncf_metainfo,
            }
        self._save_model(output_model, state_dict)

    @check_input_parameters_type()
    def export(self, export_type: ExportType, output_model: ModelEntity):
        """Export."""
        if self._compression_ctrl is None:
            super().export(export_type, output_model)
        else:
            self._compression_ctrl.prepare_for_export()
            self._model.disable_dynamic_graph_building()
            super().export(export_type, output_model)
            self._model.enable_dynamic_graph_building()

    @staticmethod
    def _load_model_data(model, data_name):
        buffer = io.BytesIO(model.get_data(data_name))
        return torch.load(buffer, map_location=torch.device("cpu"))
