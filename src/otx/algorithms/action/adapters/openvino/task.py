"""Openvino Task of OTX Action Recognition."""

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
import random
import tempfile
from typing import Any, Dict, List, Optional, Tuple, Union
from zipfile import ZipFile

import nncf
import numpy as np
import openvino.runtime as ov
from mmcv.utils import ProgressBar
from nncf.common.quantization.structs import QuantizationPreset
from openvino.model_api.adapters import OpenvinoAdapter, create_core
from openvino.model_api.models import Model

from otx.algorithms.action.adapters.openvino import (
    ActionOVClsDataLoader,
    get_ovdataloader,
    model_wrappers,
)
from otx.algorithms.action.configs.base import ActionConfig
from otx.algorithms.common.utils.ir import check_if_quantized
from otx.api.entities.annotation import AnnotationSceneEntity
from otx.api.entities.datasets import DatasetEntity, DatasetItemEntity
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
from otx.api.entities.resultset import ResultSetEntity
from otx.api.entities.subset import Subset
from otx.api.entities.task_environment import TaskEnvironment
from otx.api.serialization.label_mapper import LabelSchemaMapper, label_schema_to_bytes
from otx.api.usecases.evaluation.metrics_helper import MetricsHelper
from otx.api.usecases.exportable_code import demo
from otx.api.usecases.exportable_code.inference import BaseInferencer
from otx.api.usecases.exportable_code.prediction_to_annotation_converter import (
    ClassificationToAnnotationConverter,
    DetectionBoxToAnnotationConverter,
    IPredictionToAnnotationConverter,
)
from otx.api.usecases.tasks.interfaces.deployment_interface import IDeploymentTask
from otx.api.usecases.tasks.interfaces.evaluate_interface import IEvaluationTask
from otx.api.usecases.tasks.interfaces.inference_interface import IInferenceTask
from otx.api.usecases.tasks.interfaces.optimization_interface import (
    IOptimizationTask,
    OptimizationType,
)

logger = logging.getLogger(__name__)


# TODO: refactoring to Sphinx style.
class ActionOpenVINOInferencer(BaseInferencer):
    """ActionOpenVINOInferencer class in OpenVINO task for action recognition."""

    def __init__(
        self,
        task_type: str,
        hparams: ActionConfig,
        label_schema: LabelSchemaEntity,
        model_file: Union[str, bytes],
        weight_file: Union[str, bytes, None] = None,
        device: str = "CPU",
        num_requests: int = 1,
    ):  # pylint: disable=unused-argument, too-many-arguments
        """Inferencer implementation for OTX Action Recognition using OpenVINO backend.

        Args:
            task_type (str): Type of action task. ["ACTION_CLASSIFICATION", "ACTION_DETECTION"]
            hparams (ActionConfig): Hyperparameters for action task
            label_schema (LabelSchemaEntity): Label schema for model file
            model_file (Union[str, bytes]): XML file for model structure
            weight_file (Union[str, bytes, None]): Model weights file
                Default: None
            device (str): Device for inference. Default: "CPU"
            num_requests (int): Number of requests
        """

        self.task_type = task_type
        self.label_schema = label_schema
        model_adapter = OpenvinoAdapter(
            create_core(), model_file, weight_file, device=device, max_num_requests=num_requests
        )
        self.configuration: Dict[Any, Any] = {}
        self.model = Model.create_model(model_adapter, self.task_type, self.configuration, preload=True)
        self.converter: IPredictionToAnnotationConverter
        if self.task_type == "ACTION_CLASSIFICATION":
            self.converter = ClassificationToAnnotationConverter(self.label_schema)
        else:
            self.converter = DetectionBoxToAnnotationConverter(self.label_schema)

    def pre_process(self, image: List[DatasetItemEntity]) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Pre-process function of OpenVINO Inferencer for Action Recognition."""
        return self.model.preprocess(image)

    def post_process(
        self, prediction: Dict[str, np.ndarray], metadata: Dict[str, Any]
    ) -> Optional[AnnotationSceneEntity]:
        """Post-process function of OpenVINO Inferencer for Action Recognition."""

        prediction = self.model.postprocess(prediction, metadata)
        return self.converter.convert_to_annotation(prediction, metadata)

    def predict(self, image: List[DatasetItemEntity]) -> AnnotationSceneEntity:
        """Predict function of OpenVINO Action Inferencer for Action Recognition."""
        data, metadata = self.pre_process(image)
        raw_predictions = self.forward(data)
        predictions = self.post_process(raw_predictions, metadata)
        return predictions

    def forward(self, image: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Forward function of OpenVINO Action Inferencer for Action Recognition."""

        return self.model.infer_sync(image)


class DataLoaderWrapper:
    """DataLoader implementation for ActionOpenVINOTask."""

    def __init__(self, dataloader: Any, inferencer: BaseInferencer, shuffle: bool = True):
        self.dataloader = dataloader
        self.inferencer = inferencer
        self.shuffler = None
        if shuffle:
            self.shuffler = list(range(len(dataloader)))
            random.shuffle(self.shuffler)

    def __getitem__(self, index: int):
        """Get item from dataset."""
        if self.shuffler is not None:
            index = self.shuffler[index]

        item = self.dataloader[index]
        annotation = item[len(item) // 2].annotation_scene
        inputs, _ = self.inferencer.pre_process(item)
        return inputs, annotation

    def __len__(self):
        """Get length of dataset."""
        return len(self.dataloader)


class ActionOpenVINOTask(IDeploymentTask, IInferenceTask, IEvaluationTask, IOptimizationTask):
    """Task implementation for OTX Action Recognition using OpenVINO backend."""

    def __init__(self, task_environment: TaskEnvironment):
        self.task_environment = task_environment
        self.hparams = self.task_environment.get_hyper_parameters(ActionConfig)
        self.model = self.task_environment.model
        self.task_type = self.task_environment.model_template.task_type.name
        self.inferencer = self.load_inferencer()

    def load_inferencer(self) -> ActionOpenVINOInferencer:
        """load_inferencer function of OpenVINOTask for Action Recognition."""

        if self.model is None:
            raise RuntimeError("load_inferencer failed, model is None")

        return ActionOpenVINOInferencer(
            self.task_type,
            self.hparams,
            self.task_environment.label_schema,
            self.model.get_data("openvino.xml"),
            self.model.get_data("openvino.bin"),
        )

    # pylint: disable=no-value-for-parameter
    def infer(
        self, dataset: DatasetEntity, inference_parameters: Optional[InferenceParameters] = None
    ) -> DatasetEntity:
        """Infer function of OpenVINOTask for Action Recognition."""
        update_progress_callback = default_progress_callback
        clip_len = self.inferencer.model.t
        width = self.inferencer.model.w
        height = self.inferencer.model.h
        dataloader = get_ovdataloader(dataset, self.task_type, clip_len, width, height)
        dataset_size = len(dataloader)
        prog_bar = ProgressBar(len(dataloader))
        for i, data in enumerate(dataloader):
            prediction = self.inferencer.predict(data)
            if isinstance(dataloader, ActionOVClsDataLoader):
                dataloader.add_prediction(dataset, data, prediction)
            else:
                dataloader.add_prediction(data, prediction)
            update_progress_callback(int(i / dataset_size * 100))
            prog_bar.update()
        print("")
        return dataset

    def evaluate(self, output_resultset: ResultSetEntity, evaluation_metric: Optional[str] = None):
        """Evaluate function of OpenVINOTask."""

        if evaluation_metric is not None:
            logger.warning(f"Requested to use {evaluation_metric} metric," "but parameter is ignored.")
        if self.task_type == "ACTION_CLASSIFICATION":
            output_resultset.performance = MetricsHelper.compute_accuracy(output_resultset).get_performance()
        elif self.task_type == "ACTION_DETECTION":
            output_resultset.performance = MetricsHelper.compute_f_measure(output_resultset).get_performance()

    def deploy(self, output_model: ModelEntity) -> None:
        """Deploy function of OpenVINOTask."""

        logger.info("Deploying the model")

        work_dir = os.path.dirname(demo.__file__)
        parameters = {}  # type: Dict[Any, Any]
        parameters["type_of_model"] = f"otx_{self.task_type.lower()}"
        parameters["converter_type"] = f"{self.task_type}"
        parameters["model_parameters"] = self.inferencer.configuration
        parameters["model_parameters"]["labels"] = LabelSchemaMapper.forward(self.task_environment.label_schema)

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
        """Optimize function of OpenVINOTask."""

        if optimization_type is not OptimizationType.POT:
            raise ValueError("PTQ is the only supported optimization type for OpenVino models")

        clip_len = self.inferencer.model.t
        width = self.inferencer.model.w
        height = self.inferencer.model.h
        dataset = dataset.get_subset(Subset.TRAINING)
        data_loader = get_ovdataloader(dataset, self.task_type, clip_len, width, height)
        data_loader = DataLoaderWrapper(data_loader, self.inferencer)
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
            ov_model,
            quantization_dataset,
            subset_size=min(stat_subset_size, len(data_loader)),
            preset=preset,
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
        logger.info("PTQ optimization completed")
