"""Openvino Task of Segmentation."""

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
import json
import os
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from zipfile import ZipFile

import attr
import nncf
import numpy as np
import openvino.runtime as ov
from addict import Dict as ADDict
from nncf.common.quantization.structs import QuantizationPreset
from openvino.model_api.adapters import OpenvinoAdapter, create_core
from openvino.model_api.models import Model
from openvino.model_api.models.utils import ImageResultWithSoftPrediction

from otx.algorithms.common.utils import OTXOpenVinoDataLoader, get_default_async_reqs_num, read_py_config
from otx.algorithms.common.utils.ir import check_if_quantized
from otx.algorithms.segmentation.adapters.openvino import model_wrappers
from otx.algorithms.segmentation.configs.base import SegmentationConfig
from otx.algorithms.segmentation.utils import get_activation_map
from otx.api.entities.annotation import AnnotationSceneEntity
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.inference_parameters import (
    InferenceParameters,
    default_progress_callback,
)
from otx.api.entities.label_schema import LabelSchemaEntity
from otx.api.entities.model import (
    ModelEntity,
    ModelFormat,
    ModelOptimizationType,
    ModelPrecision,
    OptimizationMethod,
)
from otx.api.entities.optimization_parameters import OptimizationParameters
from otx.api.entities.result_media import ResultMediaEntity
from otx.api.entities.resultset import ResultSetEntity
from otx.api.entities.subset import Subset
from otx.api.entities.task_environment import TaskEnvironment
from otx.api.entities.tensor import TensorEntity
from otx.api.serialization.label_mapper import LabelSchemaMapper, label_schema_to_bytes
from otx.api.usecases.evaluation.metrics_helper import MetricsHelper
from otx.api.usecases.exportable_code import demo
from otx.api.usecases.exportable_code.inference import IInferencer
from otx.api.usecases.exportable_code.prediction_to_annotation_converter import (
    SegmentationToAnnotationConverter,
)
from otx.api.usecases.tasks.interfaces.deployment_interface import IDeploymentTask
from otx.api.usecases.tasks.interfaces.evaluate_interface import IEvaluationTask
from otx.api.usecases.tasks.interfaces.inference_interface import IInferenceTask
from otx.api.usecases.tasks.interfaces.optimization_interface import (
    IOptimizationTask,
    OptimizationType,
)
from otx.utils.logger import get_logger

logger = get_logger()


# pylint: disable=too-many-locals, too-many-statements, unused-argument
class OpenVINOSegmentationInferencer(IInferencer):
    """Inferencer implementation for Segmentation using OpenVINO backend."""

    def __init__(
        self,
        hparams: SegmentationConfig,
        label_schema: LabelSchemaEntity,
        model_file: Union[str, bytes],
        weight_file: Union[str, bytes, None] = None,
        device: str = "CPU",
        num_requests: int = 1,
    ):
        """Inferencer implementation for Segmentation using OpenVINO backend.

        :param hparams: Hyper parameters that the model should use.
        :param label_schema: LabelSchemaEntity that was used during model training.
        :param model_file: Path to model to load, `.xml`, `.bin` or `.onnx` file.
        :param device: Device to run inference on, such as CPU, GPU or MYRIAD. Defaults to "CPU".
        :param num_requests: Maximum number of requests that the inferencer can make.
            Good value is the number of available cores. Defaults to 1.
        """

        model_adapter = OpenvinoAdapter(
            create_core(),
            model_file,
            weight_file,
            device=device,
            max_num_requests=num_requests,
            plugin_config={"PERFORMANCE_HINT": "THROUGHPUT"},
        )
        self.configuration = {
            **attr.asdict(
                hparams.postprocessing,
                filter=lambda attr, value: attr.name
                not in ["header", "description", "type", "visible_in_ui", "class_name"],
            )
        }
        self.model = Model.create_model(
            model_adapter,
            "Segmentation",
            self.configuration,
            preload=True,
        )
        self.converter = SegmentationToAnnotationConverter(label_schema)
        self.callback_exceptions: List[Exception] = []
        self.model.inference_adapter.set_callback(self._async_callback)

    def predict(self, image: np.ndarray) -> Tuple[ImageResultWithSoftPrediction, AnnotationSceneEntity]:
        """Perform a prediction for a given input image."""
        result = self.model(image)
        return result, self.converter.convert_to_annotation(result)

    def enqueue_prediction(self, image: np.ndarray, id: int, result_handler: Any) -> None:
        """Runs async inference."""
        if not self.model.is_ready():
            self.model.await_any()
        image, metadata = self.model.preprocess(image)
        callback_data = id, metadata, result_handler
        self.model.inference_adapter.infer_async(image, callback_data)

    def await_all(self) -> None:
        """Await all running infer requests if any."""
        self.model.await_all()

    def _async_callback(self, request: Any, callback_args: tuple) -> None:
        """Fetches the results of async inference."""
        try:
            id, preprocessing_meta, result_handler = callback_args
            raw_prediction = self.model.inference_adapter.copy_raw_result(request)
            processed_prediciton = self.model.postprocess(raw_prediction, preprocessing_meta)
            annotation = self.converter.convert_to_annotation(processed_prediciton, preprocessing_meta)
            result_handler(id, annotation, processed_prediciton.feature_vector, processed_prediciton.saliency_map)

        except Exception as e:
            self.callback_exceptions.append(e)


class OpenVINOSegmentationTask(IDeploymentTask, IInferenceTask, IEvaluationTask, IOptimizationTask):
    """Task implementation for Segmentation using OpenVINO backend."""

    def __init__(self, task_environment: TaskEnvironment):
        self.task_environment = task_environment
        self.model = self.task_environment.model
        self.model_name = self.task_environment.model_template.model_template_id
        self.inferencer = self.load_inferencer()
        self._avg_time_per_image: Optional[float] = None

        labels = task_environment.get_labels(include_empty=False)
        self._label_dictionary = dict(enumerate(labels, 1))
        template_file_path = self.task_environment.model_template.model_template_path
        self._base_dir = os.path.abspath(os.path.dirname(template_file_path))

    @property
    def hparams(self):
        """Hparams of OpenVINO Segmentation Task."""
        return self.task_environment.get_hyper_parameters(SegmentationConfig)

    @property
    def avg_time_per_image(self) -> Optional[float]:
        """Average inference time per image."""
        return self._avg_time_per_image

    def load_inferencer(self) -> OpenVINOSegmentationInferencer:
        """load_inferencer function of OpenVINO Segmentation Task."""
        if self.model is None:
            raise RuntimeError("load_inferencer failed, model is None")
        return OpenVINOSegmentationInferencer(
            self.hparams,
            self.task_environment.label_schema,
            self.model.get_data("openvino.xml"),
            self.model.get_data("openvino.bin"),
            num_requests=get_default_async_reqs_num(),
        )

    def infer(
        self,
        dataset: DatasetEntity,
        inference_parameters: Optional[InferenceParameters] = None,
    ) -> DatasetEntity:
        """Infer function of OpenVINOSegmentationTask."""
        if inference_parameters is not None:
            update_progress_callback = inference_parameters.update_progress
            dump_soft_prediction = not inference_parameters.is_evaluation
            process_soft_prediction = inference_parameters.process_saliency_maps
            enable_async_inference = inference_parameters.enable_async_inference
        else:
            update_progress_callback = default_progress_callback
            dump_soft_prediction = True
            process_soft_prediction = False
            enable_async_inference = True

        def add_prediction(
            id: int,
            predicted_scene: AnnotationSceneEntity,
            feature_vector: Union[np.ndarray, None],
            soft_prediction: Union[np.ndarray, None],
        ):
            dataset_item = dataset[id]
            dataset_item.append_annotations(predicted_scene.annotations)

            if feature_vector is not None:
                feature_vector_media = TensorEntity(name="representation_vector", numpy=feature_vector.reshape(-1))
                dataset_item.append_metadata_item(feature_vector_media, model=self.model)

            if dump_soft_prediction and soft_prediction is not None and soft_prediction.ndim > 1:
                for label_index, label in self._label_dictionary.items():
                    current_label_soft_prediction = soft_prediction[:, :, label_index]
                    if process_soft_prediction:
                        current_label_soft_prediction = get_activation_map(
                            current_label_soft_prediction, normalize=False
                        )
                    result_media = ResultMediaEntity(
                        name=label.name,
                        type="soft_prediction",
                        label=label,
                        annotation_scene=dataset_item.annotation_scene,
                        roi=dataset_item.roi,
                        numpy=current_label_soft_prediction,
                    )
                    dataset_item.append_metadata_item(result_media, model=self.model)

        total_time = 0.0
        dataset_size = len(dataset)
        for i, dataset_item in enumerate(dataset, 1):
            start_time = time.perf_counter()
            if enable_async_inference:
                self.inferencer.enqueue_prediction(dataset_item.numpy, i - 1, add_prediction)
            else:
                result, predicted_scene = self.inferencer.predict(dataset_item.numpy)
                add_prediction(i - 1, predicted_scene, result.feature_vector, result.saliency_map)
            end_time = time.perf_counter() - start_time
            total_time += end_time

            update_progress_callback(int(i / dataset_size * 100), None)

        self.inferencer.await_all()

        self._avg_time_per_image = total_time / len(dataset)
        logger.info(f"Avg time per image: {self._avg_time_per_image} secs")
        logger.info(f"Total time: {total_time} secs")
        logger.info("Segmentation OpenVINO inference completed")

        return dataset

    def evaluate(self, output_resultset: ResultSetEntity, evaluation_metric: Optional[str] = None):
        """Evaluate function of OpenVINOSegmentationTask."""
        logger.info("Computing mDice")
        metrics = MetricsHelper.compute_dice_averaged_over_pixels(output_resultset)
        logger.info(f"mDice after evaluation: {metrics.overall_dice.value}")

        output_resultset.performance = metrics.get_performance()

    def deploy(self, output_model: ModelEntity) -> None:
        """Deploy function of OpenVINOSegmentationTask."""
        logger.info("Deploying the model")
        if self.model is None:
            raise RuntimeError("deploy failed, model is None")

        work_dir = os.path.dirname(demo.__file__)
        parameters: Dict[str, Any] = {}
        parameters["type_of_model"] = "Segmentation"
        parameters["converter_type"] = "SEGMENTATION"
        parameters["model_parameters"] = self.inferencer.configuration
        parameters["model_parameters"]["labels"] = LabelSchemaMapper.forward(self.task_environment.label_schema)

        zip_buffer = io.BytesIO()
        with ZipFile(zip_buffer, "w") as arch:
            # model files
            arch.writestr(os.path.join("model", "model.xml"), self.model.get_data("openvino.xml"))
            arch.writestr(os.path.join("model", "model.bin"), self.model.get_data("openvino.bin"))
            arch.writestr(
                os.path.join("model", "config.json"),
                json.dumps(parameters, ensure_ascii=False, indent=4),
            )
            # model_wrappers files
            for root, _, files in os.walk(os.path.dirname(model_wrappers.__file__)):
                if "__pycache__" in root:
                    continue
                for file in files:
                    file_path = os.path.join(root, file)
                    arch.write(
                        file_path,
                        os.path.join(
                            "python",
                            "model_wrappers",
                            file_path.split("model_wrappers/")[1],
                        ),
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
    ):
        """Optimize function of OpenVINOSegmentationTask."""
        logger.info("Start PTQ optimization")
        if self.model is None:
            raise RuntimeError("PTQ optimize failed, model is None")

        if optimization_type is not OptimizationType.POT:
            raise ValueError("PTQ is the only supported optimization type for OpenVino models")

        dataset = dataset.get_combined_subset([Subset.TRAINING, Subset.UNLABELED])
        data_loader = OTXOpenVinoDataLoader(dataset, self.inferencer)
        quantization_dataset = nncf.Dataset(data_loader, lambda data: data[0])

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

        optimization_config_path = os.path.join(self._base_dir, "ptq_optimization_config.py")
        ptq_config = ADDict()
        if os.path.exists(optimization_config_path):
            ptq_config = read_py_config(optimization_config_path)
        ptq_config.update(
            subset_size=min(self.hparams.pot_parameters.stat_subset_size, len(data_loader)),
            preset=QuantizationPreset(self.hparams.pot_parameters.preset.name.lower()),
        )

        compressed_model = nncf.quantize(
            ov_model,
            quantization_dataset,
            **ptq_config,
        )

        if optimization_parameters is not None:
            optimization_parameters.update_progress(90, None)

        with tempfile.TemporaryDirectory() as tempdir:
            xml_path = os.path.join(tempdir, "model.xml")
            ov.save_model(compressed_model, xml_path)
            with open(xml_path, "rb") as f:
                output_model.set_data("openvino.xml", f.read())
            with open(os.path.join(tempdir, "model.bin"), "rb") as f:
                output_model.set_data("openvino.bin", f.read())

        output_model.set_data(
            "label_schema.json",
            label_schema_to_bytes(self.task_environment.label_schema),
        )

        # set model attributes for quantized model
        output_model.model_format = ModelFormat.OPENVINO
        output_model.optimization_type = ModelOptimizationType.POT
        output_model.optimization_methods = [OptimizationMethod.QUANTIZATION]
        output_model.precision = [ModelPrecision.INT8]

        self.model = output_model
        self.inferencer = self.load_inferencer()

        if optimization_parameters is not None:
            optimization_parameters.update_progress(100, None)
        logger.info("PTQ optimization completed")
