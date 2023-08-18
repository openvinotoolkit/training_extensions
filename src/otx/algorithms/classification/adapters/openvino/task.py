"""Openvino Task of OTX Classification."""

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

import io
import json
import logging
import os
import tempfile
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union
from zipfile import ZipFile

import nncf
import numpy as np
import openvino.runtime as ov
from nncf.common.quantization.structs import QuantizationPreset
from openvino.model_api.adapters import OpenvinoAdapter, create_core
from openvino.model_api.models import Model

from otx.algorithms.classification.adapters.openvino import model_wrappers
from otx.algorithms.classification.configs import ClassificationConfig
from otx.algorithms.classification.utils import (
    get_cls_deploy_config,
    get_cls_inferencer_configuration,
    get_hierarchical_label_list,
)
from otx.algorithms.common.utils import OTXOpenVinoDataLoader
from otx.algorithms.common.utils.ir import check_if_quantized
from otx.algorithms.common.utils.utils import get_default_async_reqs_num
from otx.api.entities.annotation import AnnotationSceneEntity
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.explain_parameters import ExplainParameters
from otx.api.entities.inference_parameters import (
    InferenceParameters,
    default_progress_callback,
)
from otx.api.entities.label_schema import LabelSchemaEntity
from otx.api.entities.metadata import FloatMetadata, FloatType
from otx.api.entities.model import (
    ModelEntity,
    ModelFormat,
    ModelOptimizationType,
    ModelPrecision,
    OptimizationMethod,
)
from otx.api.entities.optimization_parameters import OptimizationParameters
from otx.api.entities.resultset import ResultSetEntity
from otx.api.entities.subset import Subset
from otx.api.entities.task_environment import TaskEnvironment
from otx.api.entities.tensor import TensorEntity
from otx.api.serialization.label_mapper import label_schema_to_bytes
from otx.api.usecases.evaluation.metrics_helper import MetricsHelper
from otx.api.usecases.exportable_code import demo
from otx.api.usecases.exportable_code.inference import BaseInferencer
from otx.api.usecases.exportable_code.prediction_to_annotation_converter import (
    ClassificationToAnnotationConverter,
)
from otx.api.usecases.tasks.interfaces.deployment_interface import IDeploymentTask
from otx.api.usecases.tasks.interfaces.evaluate_interface import IEvaluationTask
from otx.api.usecases.tasks.interfaces.explain_interface import IExplainTask
from otx.api.usecases.tasks.interfaces.inference_interface import IInferenceTask
from otx.api.usecases.tasks.interfaces.optimization_interface import (
    IOptimizationTask,
    OptimizationType,
)
from otx.api.utils.dataset_utils import add_saliency_maps_to_dataset_item

logger = logging.getLogger(__name__)


# TODO: refactoring to Sphinx style.
class ClassificationOpenVINOInferencer(BaseInferencer):
    """ClassificationOpenVINOInferencer class in OpenVINO task."""

    def __init__(
        self,
        hparams: ClassificationConfig,
        label_schema: LabelSchemaEntity,
        model_file: Union[str, bytes],
        weight_file: Union[str, bytes, None] = None,
        device: str = "CPU",
        num_requests: int = 1,
    ):  # pylint: disable=unused-argument
        """Inferencer implementation for OTXClassification using OpenVINO backend.

        :param model: Path to model to load, `.xml`, `.bin` or `.onnx` file.
        :param hparams: Hyper parameters that the model should use.
        :param num_requests: Maximum number of requests that the inferencer can make.
            Good value is the number of available cores. Defaults to 1.
        :param device: Device to run inference on, such as CPU, GPU or MYRIAD. Defaults to "CPU".
        """

        self.label_schema = label_schema
        model_adapter = OpenvinoAdapter(
            create_core(),
            model_file,
            weight_file,
            device=device,
            max_num_requests=num_requests,
            plugin_config={"PERFORMANCE_HINT": "THROUGHPUT"},
        )
        self.configuration = get_cls_inferencer_configuration(self.label_schema)
        self.model = Model.create_model(model_adapter, "otx_classification", self.configuration, preload=True)

        self.converter = ClassificationToAnnotationConverter(self.label_schema)
        self.callback_exceptions: List[Exception] = []
        self.model.inference_adapter.set_callback(self._async_callback)

    def pre_process(self, image: np.ndarray) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Pre-process function of OpenVINO Classification Inferencer."""
        return self.model.preprocess(image)

    def _async_callback(self, request: Any, callback_args: tuple) -> None:
        """Fetches the results of async inference."""
        try:
            id, preprocessing_meta, result_handler = callback_args
            prediction = self.model.inference_adapter.copy_raw_result(request)
            processed_prediciton = self.post_process(prediction, preprocessing_meta)
            aux_data = self.model.postprocess_aux_outputs(prediction, preprocessing_meta)
            result_handler(id, processed_prediciton, aux_data)

        except Exception as e:
            self.callback_exceptions.append(e)

    def post_process(
        self, prediction: Dict[str, np.ndarray], metadata: Dict[str, Any]
    ) -> Optional[AnnotationSceneEntity]:
        """Post-process function of OpenVINO Classification Inferencer."""

        classification = self.model.postprocess(prediction, metadata)
        return self.converter.convert_to_annotation(classification, metadata)

    def predict(self, image: np.ndarray) -> Tuple[AnnotationSceneEntity, np.ndarray, np.ndarray, np.ndarray, Any]:
        """Predict function of OpenVINO Classification Inferencer."""

        image, metadata = self.pre_process(image)
        raw_predictions = self.forward(image)
        predictions = self.post_process(raw_predictions, metadata)
        probs, actmap, repr_vectors, act_score = self.model.postprocess_aux_outputs(raw_predictions, metadata)

        return predictions, probs, actmap, repr_vectors, act_score

    def enqueue_prediction(self, image: np.ndarray, id: int, result_handler: Any) -> None:
        """Runs async inference."""
        if not self.model.is_ready():
            self.model.await_any()
        image, metadata = self.pre_process(image)
        callback_data = id, metadata, result_handler
        self.model.inference_adapter.infer_async(image, callback_data)

    def await_all(self) -> None:
        """Await all running infer requests if any."""
        self.model.await_all()

    def forward(self, image: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Forward function of OpenVINO Classification Inferencer."""

        return self.model.infer_sync(image)


class ClassificationOpenVINOTask(IDeploymentTask, IInferenceTask, IEvaluationTask, IExplainTask, IOptimizationTask):
    """Task implementation for OTXClassification using OpenVINO backend."""

    def __init__(self, task_environment: TaskEnvironment):
        self.task_environment = task_environment
        self.hparams = self.task_environment.get_hyper_parameters(ClassificationConfig)
        self.model = self.task_environment.model
        self.inferencer = self.load_inferencer()

    def load_inferencer(self) -> ClassificationOpenVINOInferencer:
        """load_inferencer function of ClassificationOpenVINOTask."""

        if self.model is None:
            raise RuntimeError("load_inferencer failed, model is None")

        return ClassificationOpenVINOInferencer(
            self.hparams,
            self.task_environment.label_schema,
            self.model.get_data("openvino.xml"),
            self.model.get_data("openvino.bin"),
            num_requests=get_default_async_reqs_num(),
        )

    # pylint: disable-msg=too-many-locals
    def infer(
        self, dataset: DatasetEntity, inference_parameters: Optional[InferenceParameters] = None
    ) -> DatasetEntity:
        """Infer function of ClassificationOpenVINOTask."""

        update_progress_callback = default_progress_callback
        dump_features = False
        process_saliency_maps = False
        explain_predicted_classes = True
        enable_async_inference = True

        if inference_parameters is not None:
            update_progress_callback = inference_parameters.update_progress  # type: ignore
            dump_features = not inference_parameters.is_evaluation
            process_saliency_maps = inference_parameters.process_saliency_maps
            explain_predicted_classes = inference_parameters.explain_predicted_classes
            enable_async_inference = inference_parameters.enable_async_inference

        def add_prediction(id: int, predicted_scene: AnnotationSceneEntity, aux_data: tuple):
            dataset_item = dataset[id]
            probs, saliency_map, repr_vector, act_score = aux_data
            item_labels = predicted_scene.annotations[0].get_labels()
            dataset_item.append_labels(item_labels)
            active_score_media = FloatMetadata(name="active_score", value=act_score, float_type=FloatType.ACTIVE_SCORE)
            dataset_item.append_metadata_item(active_score_media, model=self.model)

            probs_meta = TensorEntity(name="probabilities", numpy=probs.reshape(-1))
            dataset_item.append_metadata_item(probs_meta, model=self.model)

            if dump_features:
                if saliency_map is not None and repr_vector is not None:
                    feature_vec_media = TensorEntity(name="representation_vector", numpy=repr_vector.reshape(-1))
                    dataset_item.append_metadata_item(feature_vec_media, model=self.model)
                    label_list = self.task_environment.get_labels()
                    # Fix the order for hierarchical labels to adjust classes with model outputs
                    if self.inferencer.model.hierarchical:
                        label_list = get_hierarchical_label_list(
                            self.inferencer.model.hierarchical_info["cls_heads_info"], label_list
                        )

                    add_saliency_maps_to_dataset_item(
                        dataset_item=dataset_item,
                        saliency_map=saliency_map,
                        model=self.model,
                        labels=label_list,
                        predicted_scored_labels=item_labels,
                        explain_predicted_classes=explain_predicted_classes,
                        process_saliency_maps=process_saliency_maps,
                    )
                else:
                    warnings.warn(
                        "Could not find Feature Vector and Saliency Map in OpenVINO output. "
                        "Please rerun OpenVINO export or retrain the model."
                    )

        dataset_size = len(dataset)
        total_time = 0.0
        for i, dataset_item in enumerate(dataset, 1):
            start_time = time.perf_counter()
            if enable_async_inference:
                self.inferencer.enqueue_prediction(dataset_item.numpy, i - 1, add_prediction)
            else:
                predicted_scene, probs, saliency_map, repr_vector, act_score = self.inferencer.predict(
                    dataset_item.numpy
                )
                add_prediction(i - 1, predicted_scene, (probs, saliency_map, repr_vector, act_score))

            end_time = time.perf_counter() - start_time
            total_time += end_time
            update_progress_callback(int(i / dataset_size * 100))

        self.inferencer.await_all()

        logger.info(f"Avg time per image: {total_time/len(dataset)} secs")
        logger.info(f"Total time: {total_time} secs")
        logger.info("Classification OpenVINO inference completed")

        return dataset

    def explain(
        self,
        dataset: DatasetEntity,
        explain_parameters: Optional[ExplainParameters] = None,
    ) -> DatasetEntity:
        """Explain function of ClassificationOpenVINOTask."""

        update_progress_callback = default_progress_callback
        process_saliency_maps = False
        explain_predicted_classes = True
        if explain_parameters is not None:
            update_progress_callback = explain_parameters.update_progress  # type: ignore
            process_saliency_maps = explain_parameters.process_saliency_maps
            explain_predicted_classes = explain_parameters.explain_predicted_classes

        dataset_size = len(dataset)
        label_list = self.task_environment.get_labels()
        # Fix the order for hierarchical labels to adjust classes with model outputs
        if self.inferencer.model.hierarchical:
            label_list = get_hierarchical_label_list(
                self.inferencer.model.hierarchical_info["cls_heads_info"], label_list
            )
        for i, dataset_item in enumerate(dataset, 1):
            predicted_scene, _, saliency_map, _, _ = self.inferencer.predict(dataset_item.numpy)
            if saliency_map is None:
                raise RuntimeError(
                    "There is no Saliency Map in OpenVINO IR model output. "
                    "Please export model to OpenVINO IR with dump_features"
                )

            item_labels = predicted_scene.annotations[0].get_labels()
            dataset_item.append_labels(item_labels)
            add_saliency_maps_to_dataset_item(
                dataset_item=dataset_item,
                saliency_map=saliency_map,
                model=self.model,
                labels=label_list,
                predicted_scored_labels=item_labels,
                explain_predicted_classes=explain_predicted_classes,
                process_saliency_maps=process_saliency_maps,
            )
            update_progress_callback(int(i / dataset_size * 100))
        return dataset

    def evaluate(self, output_resultset: ResultSetEntity, evaluation_metric: Optional[str] = None):
        """Evaluate function of ClassificationOpenVINOTask."""

        if evaluation_metric is not None:
            logger.warning(
                f"Requested to use {evaluation_metric} metric," "but parameter is ignored. Use accuracy instead."
            )
        output_resultset.performance = MetricsHelper.compute_accuracy(output_resultset).get_performance()

    def deploy(self, output_model: ModelEntity) -> None:
        """Deploy function of ClassificationOpenVINOTask."""

        logger.info("Deploying the model")

        work_dir = os.path.dirname(demo.__file__)
        parameters = get_cls_deploy_config(self.task_environment.label_schema, self.inferencer.configuration)

        if self.model is None:
            raise RuntimeError("deploy failed, model is None")

        zip_buffer = io.BytesIO()
        with ZipFile(zip_buffer, "w") as arch:
            # model files
            arch.writestr(os.path.join("model", "model.xml"), self.model.get_data("openvino.xml"))
            arch.writestr(os.path.join("model", "model.bin"), self.model.get_data("openvino.bin"))
            arch.writestr(os.path.join("model", "config.json"), json.dumps(parameters, ensure_ascii=False, indent=4))
            # model_wrappers files
            for root, _, files in os.walk(os.path.dirname(model_wrappers.__file__)):
                if "__pycache__" in root:
                    continue
                for file in files:
                    file_path = os.path.join(root, file)
                    arch.write(
                        file_path, os.path.join("python", "model_wrappers", file_path.split("model_wrappers/")[1])
                    )
            # other python files
            arch.write(os.path.join(work_dir, "requirements.txt"), os.path.join("python", "requirements.txt"))
            arch.write(os.path.join(work_dir, "LICENSE"), os.path.join("python", "LICENSE"))
            arch.write(os.path.join(work_dir, "demo.py"), os.path.join("python", "demo.py"))
            arch.write(os.path.join(work_dir, "README.md"), os.path.join(".", "README.md"))
        output_model.exportable_code = zip_buffer.getvalue()
        logger.info("Deploying completed")

    def optimize(
        self,
        optimization_type: OptimizationType,
        dataset: DatasetEntity,
        output_model: ModelEntity,
        optimization_parameters: Optional[OptimizationParameters] = None,
    ):  # pylint: disable=too-many-locals
        """Optimize function of ClassificationOpenVINOTask."""

        if optimization_type is not OptimizationType.POT:
            raise ValueError("PTQ is the only supported optimization type for OpenVino models")

        dataset = dataset.get_subset(Subset.TRAINING)
        data_loader = OTXOpenVinoDataLoader(dataset, self.inferencer)

        quantization_dataset = nncf.Dataset(data_loader, lambda data: data[0])

        if self.model is None:
            raise RuntimeError("optimize failed, model is None")

        with tempfile.TemporaryDirectory() as tempdir:
            xml_path = os.path.join(tempdir, "model.xml")
            bin_path = os.path.join(tempdir, "model.bin")
            with open(xml_path, "wb") as f:
                f.write(self.model.get_data("openvino.xml"))
            with open(bin_path, "wb") as f:
                f.write(self.model.get_data("openvino.bin"))

            ov_model = ov.Core().read_model(xml_path)
            if check_if_quantized(ov_model):
                raise RuntimeError("Model is already optimized by PTQ")

        if optimization_parameters is not None:
            optimization_parameters.update_progress(10, None)

        stat_subset_size = self.hparams.pot_parameters.stat_subset_size
        preset = QuantizationPreset(self.hparams.pot_parameters.preset.name.lower())

        compressed_model = nncf.quantize(
            ov_model, quantization_dataset, subset_size=min(stat_subset_size, len(data_loader)), preset=preset
        )

        if optimization_parameters is not None:
            optimization_parameters.update_progress(90, None)

        with tempfile.TemporaryDirectory() as tempdir:
            xml_path = os.path.join(tempdir, "model.xml")
            ov.serialize(compressed_model, xml_path)
            with open(xml_path, "rb") as f:
                output_model.set_data("openvino.xml", f.read())
            with open(os.path.join(tempdir, "model.bin"), "rb") as f:
                output_model.set_data("openvino.bin", f.read())

        output_model.set_data("label_schema.json", label_schema_to_bytes(self.task_environment.label_schema))

        # set model attributes for quantized model
        output_model.model_format = ModelFormat.OPENVINO
        output_model.optimization_type = ModelOptimizationType.POT
        output_model.optimization_methods = [OptimizationMethod.QUANTIZATION]
        output_model.precision = [ModelPrecision.INT8]

        self.model = output_model
        self.inferencer = self.load_inferencer()

        if optimization_parameters is not None:
            optimization_parameters.update_progress(100, None)
        logger.info("PQT optimization completed")
