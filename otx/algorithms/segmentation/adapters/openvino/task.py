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
from typing import Any, Dict, Optional, Tuple, Union
from zipfile import ZipFile

import attr
import numpy as np
from addict import Dict as ADDict
from compression.api import DataLoader
from compression.engines.ie_engine import IEEngine
from compression.graph import load_model, save_model
from compression.graph.model_utils import compress_model_weights, get_nodes_by_type
from compression.pipeline.initializer import create_pipeline
from openvino.model_zoo.model_api.adapters import OpenvinoAdapter, create_core
from openvino.model_zoo.model_api.models import Model

from otx.algorithms.common.utils.logger import get_logger
from otx.algorithms.segmentation.adapters.openvino import model_wrappers
from otx.algorithms.segmentation.adapters.openvino.model_wrappers.blur import (
    get_activation_map,
)
from otx.algorithms.segmentation.configs.base import SegmentationConfig
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
from otx.api.usecases.exportable_code.inference import BaseInferencer
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
from otx.api.utils.argument_checks import (
    DatasetParamTypeCheck,
    check_input_parameters_type,
)

logger = get_logger()


# pylint: disable=too-many-locals, too-many-statements, unused-argument
class OpenVINOSegmentationInferencer(BaseInferencer):
    """Inferencer implementation for Segmentation using OpenVINO backend."""

    @check_input_parameters_type()
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
            create_core(), model_file, weight_file, device=device, max_num_requests=num_requests
        )
        self.configuration = {
            **attr.asdict(
                hparams.postprocessing,
                filter=lambda attr, value: attr.name
                not in ["header", "description", "type", "visible_in_ui", "class_name"],
            )
        }
        self.model = Model.create_model(
            hparams.postprocessing.class_name.value, model_adapter, self.configuration, preload=True
        )
        self.converter = SegmentationToAnnotationConverter(label_schema)

    @check_input_parameters_type()
    def pre_process(self, image: np.ndarray) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Pre-process function of OpenVINO Segmentation Inferencer."""
        return self.model.preprocess(image)

    @check_input_parameters_type()
    def post_process(
        self, prediction: Dict[str, np.ndarray], metadata: Dict[str, Any]
    ) -> Tuple[AnnotationSceneEntity, Any, Any]:
        """Post-process function of OpenVINO Segmentation Inferencer."""
        hard_prediction = self.model.postprocess(prediction, metadata)
        soft_prediction = metadata["soft_prediction"]
        feature_vector = metadata["feature_vector"]
        predicted_scene = self.converter.convert_to_annotation(hard_prediction, metadata)

        return predicted_scene, feature_vector, soft_prediction

    @check_input_parameters_type()
    def predict(self, image: np.ndarray) -> Tuple[AnnotationSceneEntity, Any, Any]:
        """Perform a prediction for a given input image."""
        image, metadata = self.pre_process(image)
        predictions = self.forward(image)
        predictions = self.post_process(predictions, metadata)
        return predictions

    @check_input_parameters_type()
    def forward(self, image: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Forward function of OpenVINO Segmentation Inferencer."""
        return self.model.infer_sync(image)


class OTXOpenVinoDataLoader(DataLoader):
    """Data loader for OTXDetection using OpenVINO backend."""

    @check_input_parameters_type({"dataset": DatasetParamTypeCheck})
    def __init__(self, dataset: DatasetEntity, inferencer: BaseInferencer):
        self.dataset = dataset
        self.inferencer = inferencer

    @check_input_parameters_type()
    def __getitem__(self, index: int):
        """Return dataset item from index."""
        image = self.dataset[index].numpy
        annotation = self.dataset[index].annotation_scene
        inputs, metadata = self.inferencer.pre_process(image)

        return (index, annotation), inputs, metadata

    def __len__(self):
        """Length of OTXOpenVinoDataLoader."""
        return len(self.dataset)


class OpenVINOSegmentationTask(IDeploymentTask, IInferenceTask, IEvaluationTask, IOptimizationTask):
    """Task implementation for Segmentation using OpenVINO backend."""

    @check_input_parameters_type()
    def __init__(self, task_environment: TaskEnvironment):
        self.task_environment = task_environment
        self.model = self.task_environment.model
        self.model_name = self.task_environment.model_template.model_template_id
        self.inferencer = self.load_inferencer()

        labels = task_environment.get_labels(include_empty=False)
        self._label_dictionary = dict(enumerate(labels, 1))
        template_file_path = self.task_environment.model_template.model_template_path
        self._base_dir = os.path.abspath(os.path.dirname(template_file_path))

    @property
    def hparams(self):
        """Hparams of OpenVINO Segmentation Task."""
        return self.task_environment.get_hyper_parameters(SegmentationConfig)

    def load_inferencer(self) -> OpenVINOSegmentationInferencer:
        """load_inferencer function of OpenVINO Segmentation Task."""
        if self.model is None:
            raise RuntimeError("load_inferencer failed, model is None")
        return OpenVINOSegmentationInferencer(
            self.hparams,
            self.task_environment.label_schema,
            self.model.get_data("openvino.xml"),
            self.model.get_data("openvino.bin"),
        )

    @check_input_parameters_type({"dataset": DatasetParamTypeCheck})
    def infer(
        self, dataset: DatasetEntity, inference_parameters: Optional[InferenceParameters] = None
    ) -> DatasetEntity:
        """Infer function of OpenVINOSegmentationTask."""
        if inference_parameters is not None:
            update_progress_callback = inference_parameters.update_progress
            dump_soft_prediction = not inference_parameters.is_evaluation
        else:
            update_progress_callback = default_progress_callback
            dump_soft_prediction = True

        dataset_size = len(dataset)
        for i, dataset_item in enumerate(dataset, 1):
            predicted_scene, feature_vector, soft_prediction = self.inferencer.predict(dataset_item.numpy)
            dataset_item.append_annotations(predicted_scene.annotations)

            if feature_vector is not None:
                feature_vector_media = TensorEntity(name="representation_vector", numpy=feature_vector.reshape(-1))
                dataset_item.append_metadata_item(feature_vector_media, model=self.model)

            if dump_soft_prediction:
                for label_index, label in self._label_dictionary.items():
                    if label_index == 0:
                        continue
                    current_label_soft_prediction = soft_prediction[:, :, label_index]
                    class_act_map = get_activation_map(current_label_soft_prediction)
                    result_media = ResultMediaEntity(
                        name=label.name,
                        type="soft_prediction",
                        label=label,
                        annotation_scene=dataset_item.annotation_scene,
                        roi=dataset_item.roi,
                        numpy=class_act_map,
                    )
                    dataset_item.append_metadata_item(result_media, model=self.model)

            update_progress_callback(int(i / dataset_size * 100), None)

        return dataset

    @check_input_parameters_type()
    def evaluate(self, output_resultset: ResultSetEntity, evaluation_metric: Optional[str] = None):
        """Evaluate function of OpenVINOSegmentationTask."""
        logger.info("Computing mDice")
        metrics = MetricsHelper.compute_dice_averaged_over_pixels(output_resultset)
        logger.info(f"mDice after evaluation: {metrics.overall_dice.value}")

        output_resultset.performance = metrics.get_performance()

    @check_input_parameters_type()
    def deploy(self, output_model: ModelEntity) -> None:
        """Deploy function of OpenVINOSegmentationTask."""
        logger.info("Deploying the model")
        if self.model is None:
            raise RuntimeError("deploy failed, model is None")

        work_dir = os.path.dirname(demo.__file__)
        parameters = {}
        parameters["type_of_model"] = self.hparams.postprocessing.class_name.value
        parameters["converter_type"] = "SEGMENTATION"
        parameters["model_parameters"] = self.inferencer.configuration
        parameters["model_parameters"]["labels"] = LabelSchemaMapper.forward(self.task_environment.label_schema)

        zip_buffer = io.BytesIO()
        with ZipFile(zip_buffer, "w") as arch:
            # model files
            arch.writestr(os.path.join("model", "model.xml"), self.model.get_data("openvino.xml"))
            arch.writestr(os.path.join("model", "model.bin"), self.model.get_data("openvino.bin"))
            arch.writestr(os.path.join("model", "config.json"), json.dumps(parameters, ensure_ascii=False, indent=4))
            # model_wrappers files
            for root, _, files in os.walk(os.path.dirname(model_wrappers.__file__)):
                for file in files:
                    file_path = os.path.join(root, file)
                    arch.write(
                        file_path, os.path.join("python", "model_wrappers", file_path.split("model_wrappers/")[1])
                    )
            # other python files
            arch.write(os.path.join(work_dir, "requirements.txt"), os.path.join("python", "requirements.txt"))
            arch.write(os.path.join(work_dir, "LICENSE"), os.path.join("python", "LICENSE"))
            arch.write(os.path.join(work_dir, "README.md"), os.path.join("python", "README.md"))
            arch.write(os.path.join(work_dir, "demo.py"), os.path.join("python", "demo.py"))
        output_model.exportable_code = zip_buffer.getvalue()
        logger.info("Deploying completed")

    @check_input_parameters_type({"dataset": DatasetParamTypeCheck})
    def optimize(
        self,
        optimization_type: OptimizationType,
        dataset: DatasetEntity,
        output_model: ModelEntity,
        optimization_parameters: Optional[OptimizationParameters] = None,
    ):
        """Optimize function of OpenVINOSegmentationTask."""
        logger.info("Start POT optimization")
        if self.model is None:
            raise RuntimeError("POT optimize failed, model is None")

        if optimization_type is not OptimizationType.POT:
            raise ValueError("POT is the only supported optimization type for OpenVino models")

        dataset = dataset.get_subset(Subset.TRAINING)
        data_loader = OTXOpenVinoDataLoader(dataset, self.inferencer)

        with tempfile.TemporaryDirectory() as tempdir:
            xml_path = os.path.join(tempdir, "model.xml")
            bin_path = os.path.join(tempdir, "model.bin")
            with open(xml_path, "wb") as f:
                f.write(self.model.get_data("openvino.xml"))
            with open(bin_path, "wb") as f:
                f.write(self.model.get_data("openvino.bin"))

            model_config = ADDict({"model_name": "openvino_model", "model": xml_path, "weights": bin_path})

            model = load_model(model_config)

            if get_nodes_by_type(model, ["FakeQuantize"]):
                raise RuntimeError("Model is already optimized by POT")

        if optimization_parameters is not None:
            optimization_parameters.update_progress(10, None)

        engine_config = ADDict({"device": "CPU"})

        optimization_config_path = os.path.join(self._base_dir, "pot_optimization_config.json")
        if os.path.exists(optimization_config_path):
            with open(optimization_config_path, encoding="UTF-8") as f_src:
                algorithms = ADDict(json.load(f_src))["algorithms"]
        else:
            algorithms = [ADDict({"name": "DefaultQuantization", "params": {"target_device": "ANY"}})]
        for algo in algorithms:
            algo.params.stat_subset_size = self.hparams.pot_parameters.stat_subset_size
            algo.params.shuffle_data = True
            if "Quantization" in algo["name"]:
                algo.params.preset = self.hparams.pot_parameters.preset.name.lower()

        engine = IEEngine(config=engine_config, data_loader=data_loader, metric=None)

        pipeline = create_pipeline(algorithms, engine)

        compressed_model = pipeline.run(model)

        compress_model_weights(compressed_model)

        if optimization_parameters is not None:
            optimization_parameters.update_progress(90, None)

        with tempfile.TemporaryDirectory() as tempdir:
            save_model(compressed_model, tempdir, model_name="model")
            with open(os.path.join(tempdir, "model.xml"), "rb") as f:
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
        logger.info("POT optimization completed")
